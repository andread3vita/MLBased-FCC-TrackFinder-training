import torch
import torch.nn as nn
import os
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import torch.distributed as dist
import numpy as np
from src.logger.logger_wandb import log_losses_wandb_tracking
from src.layers.inference_oc_tracks import (
    evaluate_efficiency_tracks,
    store_at_batch_end,
    store_at_batch_end_hits,
    get_clustering
)
from src.layers.losses import object_condensation_loss_tracking
from src.layers.batch_operations import obtain_batch_numbers

from src.gatr_v111.nets.gatr import GATr
from src.gatr_v111.layers.attention.config import SelfAttentionConfig
from src.gatr_v111.layers.mlp.config import MLPConfig
from src.gatr_v111.interface import (
    embed_point,
    embed_scalar,
    embed_translation,
)
from xformers.ops.fmha import BlockDiagonalMask
from src.gatr_v111.primitives.invariants import compute_inner_product_mask
from src.gatr_v111.primitives.linear import _compute_pin_equi_linear_basis
from src.gatr_v111.primitives.attention import _build_dist_basis

import wandb
from src.logger.plotting_tools import (PlotCoordinates, efficiency_purity_plot, trackingEfficiencyPlot)
import pandas as pd

class ExampleWrapper(L.LightningModule): 
    def __init__(
        self,
        args,
    ):
        super().__init__()
        blocks = 10
        hidden_mv_channels = 16
        hidden_s_channels = 64
        self.input_dim = 3
        self.output_dim = 4
        self.args = args
        self.basis_gp = None
        self.basis_outer = None
        self.pin_basis = None
        self.basis_q = None
        self.basis_k = None
        self.basis_gp_mask = None
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.1)


        self.load_basis()
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
            basis_gp=self.basis_gp,
            basis_outer=self.basis_outer,
            basis_pin=self.pin_basis,
            basis_q=self.basis_q,
            basis_k=self.basis_k,
            basis_gp_mask = self.basis_gp_mask, 
        )

        self.clustering = nn.Linear(16, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(16, 1)
        self.vector_like_data = True
        self.df_batch_buffer = []

    def load_basis(self):

        filename = "gatr_utils/geometric_product.pt"
        sparse_basis = torch.load(filename).to(torch.float32)
        basis = sparse_basis.to_dense()
        self.basis_gp = basis.to(device="cuda")
        filename = "gatr_utils/outer_product.pt"
        sparse_basis_outer = torch.load(filename).to(torch.float32)
        sparse_basis_outer = sparse_basis_outer.to_dense()
        self.basis_outer = sparse_basis_outer.to(device="cuda")

        self.pin_basis = _compute_pin_equi_linear_basis(
            device=self.basis_gp.device, dtype=basis.dtype
        )
        self.basis_q, self.basis_k = _build_dist_basis(
            device=self.basis_gp.device, dtype=basis.dtype
        )
        mask = compute_inner_product_mask(self.basis_gp, device=self.basis_gp.device)
        columns = torch.arange(0, 16).to(self.basis_gp.device)
        colums_take = columns[mask.bool()]
        self.basis_gp_mask = colums_take

    def forward(self, g, input):  
        
        pos_hits_xyz = input[:, 0:3]
        hit_type = input[:, 3].view(-1, 1)
        vector = input[:, 4:]
        
        inputs = self.ScaledGooeyBatchNorm2_1(pos_hits_xyz)
        embedded_inputs = embed_point(inputs) + embed_scalar(hit_type) + embed_translation(vector)
        # embedded_inputs = embed_point(inputs) + embed_scalar(hit_type)


        embedded_inputs = embedded_inputs.unsqueeze(-2)
        scalars = torch.zeros((inputs.shape[0], 1), device=inputs.device, dtype=inputs.dtype)
        mask = self.build_attention_mask(g)
        
        embedded_outputs, _ = self.gatr(
            embedded_inputs, scalars=scalars, attention_mask=mask
        )
        output = embedded_outputs[:, 0, :]
        x_cluster_coord = self.clustering(output)
        beta = self.beta(output)
        x = torch.cat((x_cluster_coord, beta), dim=1)

        return x

    def build_attention_mask(self, g):
        """Construct attention mask from pytorch geometric batch.

        Parameters
            ----------
            inputs : torch_geometric.data.Batch
            Data batch.

        Returns
        -------
        attention_mask : xformers.ops.fmha.BlockDiagonalMask
            Block-diagonal attention mask: within each sample, each token can attend to each other
            token.
        """
        batch_numbers = obtain_batch_numbers(g)
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]

        pos_hits_xyz = batch_g.ndata["pos_hits_xyz"]
        hit_type = batch_g.ndata["hit_type"].view(-1, 1)
        vector = batch_g.ndata["vector"]
        input_ = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)
        
        model_output = self(batch_g, input_)

        x_cluster_coord = model_output[:, :-1]
        beta = model_output[:, -1]

        # labels = get_clustering(beta, x_cluster_coord, 0.5, 0.1)
        # unique_labels, clustering = torch.unique(labels, return_inverse=True)
        # if unique_labels[0] != -1:
        #     clustering += 1
        # labels = clustering.long()

        (loss, losses) = object_condensation_loss_tracking(
            batch_g,
            model_output,
            y,
            clust_loss_only=True,
            add_energy_loss=False,
            calc_e_frac_loss=False,
            q_min=self.args.qmin,
            frac_clustering_loss=self.args.frac_cluster_loss,
            attr_weight=self.args.L_attractive_weight,
            repul_weight=self.args.L_repulsive_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            loss_type= self.args.loss_type,
            tracking=True,
        )
                
        if torch.isnan(loss):
            print(f"Batch {batch_idx} returns NaN, skip.")
            return None
        
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss)

            # if batch_idx % 1000 == 0:

            #     batch_g.ndata["final_coords"] = batch_g.ndata["pos_hits_xyz"]
            #     batch_g.ndata["reco_labels"] = labels
            #     fig1 = PlotCoordinates(
            #         batch_g,
            #         path="final_coords",
            #         outdir=self.args.model_prefix,
            #         epoch=self.current_epoch,
            #         step_count=self.global_step,
            #     )

            #     batch_g.ndata["embedded_coords"] = x_cluster_coord
            #     batch_g.ndata["beta"] = beta
            #     fig2 = PlotCoordinates(
            #         batch_g,
            #         path="embedded_coords",
            #         outdir=self.args.model_prefix,
            #         epoch=self.current_epoch,
            #         step_count=self.global_step,
            #     )

            #     batch_g.ndata["original_coords"] = batch_g.ndata["pos_hits_xyz"]
            #     fig3 = PlotCoordinates(
            #         batch_g,
            #         path="input_coords",
            #         outdir=self.args.model_prefix,
            #         epoch=self.current_epoch,
            #         step_count=self.global_step,
            #     )

               
            #     html_string1 = fig1.to_html(full_html=False, auto_play=False)
            #     html_string2 = fig2.to_html(full_html=False, auto_play=False)
            #     html_string3 = fig3.to_html(full_html=False, auto_play=False)


            #     wandb.log({"final_coords": wandb.Html(html_string1),
            #                "embedding_coords" :wandb.Html(html_string2),
            #                "true_coords": wandb.Html(html_string3)})

        return loss

    def validation_step(self, batch, batch_idx):

        y = batch[1]
        batch_g = batch[0]

        pos_hits_xyz = batch_g.ndata["pos_hits_xyz"]
        hit_type = batch_g.ndata["hit_type"].view(-1, 1)
        vector = batch_g.ndata["vector"]

        input_ = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)

        model_output = self(batch_g, input_)

        # import plotly.graph_objects as go
        # from sklearn.decomposition import PCA

        # # Assume model_output is your Nx7 tensor
        # # model_output = self(batch_g, input_)

        # # Example: create dummy data if you don't have the model output yet
        # # model_output = torch.randn(100, 7)

        # # Extract components
        # features_6d = model_output[:, :6].detach().cpu().numpy()  # Nx6
        # color_feature = model_output[:, 6].detach().cpu().numpy()  # N
        # color_feature = torch.sigmoid(model_output[:, 6]).detach().cpu().numpy()

        # # Project 6D -> 3D using PCA
        # pca = PCA(n_components=3)
        # coords_3d = pca.fit_transform(features_6d)  # Nx3

        # x, y, z = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]

        # # Create interactive 3D scatter plot
        # fig = go.Figure(data=[go.Scatter3d(
        #     x=x,
        #     y=y,
        #     z=z,
        #     mode='markers',
        #     marker=dict(
        #         size=5,
        #         color=color_feature,
        #         colorscale='Viridis',
        #         colorbar=dict(title='7th Feature'),
        #         opacity=0.85,
        #         line=dict(width=0.5, color='white')
        #     ),
        #     text=[f'Feature: {v:.4f}' for v in color_feature],
        #     hovertemplate=(
        #         '<b>x:</b> %{x:.3f}<br>'
        #         '<b>y:</b> %{y:.3f}<br>'
        #         '<b>z:</b> %{z:.3f}<br>'
        #         '<b>7th feature:</b> %{text}<extra></extra>'
        #     )
        # )])

        # fig.update_layout(
        #     title=dict(text='3D PCA Projection of Model Output', x=0.5),
        #     scene=dict(
        #         xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
        #         yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
        #         zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
        #         bgcolor='rgb(10, 10, 20)',
        #         xaxis=dict(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.2)'),
        #         yaxis=dict(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.2)'),
        #         zaxis=dict(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.2)'),
        #     ),
        #     paper_bgcolor='rgb(15, 15, 25)',
        #     font=dict(color='white'),
        #     margin=dict(l=0, r=0, b=0, t=40),
        # )

        # # Add explained variance annotation
        # total_var = pca.explained_variance_ratio_[:3].sum() * 100
        # fig.add_annotation(
        #     text=f'Total variance explained: {total_var:.1f}%',
        #     xref='paper', yref='paper',
        #     x=0.01, y=0.01,
        #     showarrow=False,
        #     font=dict(size=12, color='lightgray')
        # )

        # fig.write_html('model_output_3d.html')
        # print(f"Saved to model_output_3d.html")
        # print(f"Points: {len(x)} | Variance explained: {total_var:.1f}%")

        # sys.exit()
        batch_g.ndata["model_output"] = model_output

        (loss, losses) = object_condensation_loss_tracking(
            batch_g,
            model_output,
            y,
            clust_loss_only=True,
            add_energy_loss=False,
            calc_e_frac_loss=False,
            q_min=self.args.qmin,
            frac_clustering_loss=self.args.frac_cluster_loss,
            attr_weight=self.args.L_attractive_weight,
            repul_weight=self.args.L_repulsive_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            loss_type=self.args.loss_type,
            tracking=True,
        )

        if self.trainer.is_global_zero:
                
                # x_cluster_coord = model_output[:, :3]
                # beta = model_output[:, 3]
                # beta = torch.sigmoid(beta)

                # # print("shape:", beta.shape)

                # # print("mean:", beta.mean().item())
                # # print("std:", beta.std().item())
                # # print("min:", beta.min().item())
                # # print("max:", beta.max().item())

                # labels = get_clustering(beta, x_cluster_coord, 0.5, 0.1)
                # unique_labels, clustering = torch.unique(labels, return_inverse=True)
                # if unique_labels[0] != -1:
                #     clustering += 1
                # labels = clustering.long()

                # batch_g.ndata["original_coords"] = batch_g.ndata["pos_hits_xyz"]
                # batch_g.ndata["reco_labels"] = labels
                # fig1 = PlotCoordinates(
                #     batch_g,
                #     path="final_coords",
                #     outdir=self.args.model_prefix,
                #     epoch=self.current_epoch,
                #     step_count=self.global_step,
                # )

                # batch_g.ndata["embedded_coords"] = x_cluster_coord
                # batch_g.ndata["beta"] = beta
                # fig2 = PlotCoordinates(
                #     batch_g,
                #     path="embedded_coords",
                #     outdir=self.args.model_prefix,
                #     epoch=self.current_epoch,
                #     step_count=self.global_step,
                # )

                # fig3 = PlotCoordinates(
                #     batch_g,
                #     path="input_coords",
                #     outdir=self.args.model_prefix,
                #     epoch=self.current_epoch,
                #     step_count=self.global_step,
                # )

                # fig1.write_html("fig1.html")
                # fig2.write_html("fig2.html")
                # fig3.write_html("fig3.html")


                # print(batch_g.ndata["pos_hits_xyz"].shape)
                # print(batch_g.ndata["pos_hits_xyz"].shape)
                # print(batch_g.ndata["beta"].shape)

                # sys.exit()

                log_losses_wandb_tracking(True, self.global_step, 0, losses, loss, val=True)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        
        # part_keys = [
        # "part_theta",    # 0
        # "part_phi",      # 1
        # "part_m",        # 2
        # "part_pid",      # 3
        # "part_id",       # 4
        # "part_p",        # 5
        # "part_p_t",      # 6
        # "gen_status",    # 7
        # "part_parent",   # 8
        # "batch_id"       # 9
        # ]
        # partInfo = {key: y[:, i] for i, key in enumerate(part_keys)}

        # # =========================
        # # Convert to pandas DataFrame
        # # =========================
        # df_part = pd.DataFrame({
        #     key: tensor.detach().cpu().numpy()
        #     for key, tensor in partInfo.items()
        # })

        # # =========================
        # # Save dataframe as .pt
        # # =========================
        # torch.save(df_part, "particle_info.pt")


        # # move to CPU if needed
        # model_output = model_output.detach().cpu()
        # ndata = batch_g.ndata

        # df = pd.DataFrame({
        #     "fileNumber": ndata["fileNumber"].detach().cpu().numpy(),
        #     "eventNumber": ndata["eventNumber"].detach().cpu().numpy(),

        #     "pos_hits_xyz.x": ndata["pos_hits_xyz"][:, 0].detach().cpu().numpy(),
        #     "pos_hits_xyz.y": ndata["pos_hits_xyz"][:, 1].detach().cpu().numpy(),
        #     "pos_hits_xyz.z": ndata["pos_hits_xyz"][:, 2].detach().cpu().numpy(),

        #     "hit_type": ndata["hit_type"].detach().cpu().numpy(),
        #     "part_id": ndata["particle_number_nomap"].detach().cpu().numpy(),

        #     # keep full vector as a list-column
        #     "model_output": [row.tolist() for row in model_output]
        # })

        # torch.save(df, "graph_dataframe.pt")

        # sys.exit()

        if self.trainer.is_global_zero and self.args.predict:

            df_batch, df_hits = evaluate_efficiency_tracks(
                batch_g,
                model_output,
                y,
                0,
                batch_idx,
                0,
                path_save=self.args.model_prefix + "showers_df_evaluation",
                store=True,
                predict=False,
                tau=self.args.tau

            )
            
            if self.args.predict:
                if len(df_batch) > 0:
                    self.df_showers.append(df_batch)
                    
                if len(df_hits) > 0:
                    self.df_showers_hits.append(df_hits)

            store_at_batch_end(
                    self.args.model_prefix + "showers_df_evaluation",
                    self.df_showers,
                    0,
                    0,
                    0,
                    predict=True,
                )
                
            store_at_batch_end_hits(
                    self.args.model_prefix + "showers_df_evaluation",
                    self.df_showers_hits,
                    0,
                    0,
                    0,
                    predict=True,
                )       

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.df_batch_buffer = []
        self.df_showers = []
        self.df_showers_hits = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 2 or self.args.predict:
            self.ScaledGooeyBatchNorm2_1.momentum = 0

    def on_validation_epoch_end(self):

        if self.args.predict:
            store_at_batch_end(
                self.args.model_prefix + "showers_df_evaluation",
                self.df_showers,
                0,
                0,
                0,
                predict=True,
            )
            
            store_at_batch_end_hits(
                self.args.model_prefix + "showers_df_evaluation",
                self.df_showers_hits,
                0,
                0,
                0,
                predict=True,
            )

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.start_lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            threshold=1e-3,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }