import torch
import torch.nn as nn
import os
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.logger.logger_wandb import log_losses_wandb_tracking
from src.layers.inference_oc_tracks import (
    evaluate_efficiency_tracks,
    store_at_batch_end,
    store_at_batch_end_hits
)
from src.layers.losses import object_condensation_loss_tracking
from src.layers.batch_operations import obtain_batch_numbers

from src.gatr_v111_onnx.nets.gatr import GATr
from src.gatr_v111_onnx.layers.attention.config import SelfAttentionConfig
from src.gatr_v111_onnx.layers.mlp.config import MLPConfig
from src.gatr_v111_onnx.interface import (
    embed_point,
    embed_scalar,
    embed_translation,
)
from src.gatr_v111_onnx.primitives.invariants import   compute_inner_product_mask
from src.gatr_v111_onnx.primitives.linear import _compute_pin_equi_linear_basis
from src.gatr_v111_onnx.primitives.attention import _build_dist_basis

class ExampleWrapper(L.LightningModule):  # nn.Module L.LightningModule
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

    def load_basis(self):

        this_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(this_dir, "../../"))
        
        filename = os.path.join(parent_dir, "gatr_utils/geometric_product.pt")
        sparse_basis = torch.load(filename).to(torch.float32)
        basis = sparse_basis.to_dense()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.basis_gp = basis.to(device=device)
        
        filename = os.path.join(parent_dir, "gatr_utils/outer_product.pt")
        sparse_basis_outer = torch.load(filename).to(torch.float32)
        sparse_basis_outer = sparse_basis_outer.to_dense()
        self.basis_outer = sparse_basis_outer.to(device=device)

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

    def forward(self, input):  
        
        pos_hits_xyz = input[:, 0:3]
        hit_type = input[:, 3].view(-1, 1)
        vector = input[:, 4:]
        
        inputs = self.ScaledGooeyBatchNorm2_1(pos_hits_xyz)
        embedded_inputs = embed_point(inputs) + embed_scalar(hit_type) + embed_translation(vector)
        embedded_inputs = embedded_inputs.unsqueeze(-2)
        
        # with open("inputs_dump.txt", "w") as f:
        #     for i, inp in enumerate(input):
        #         arr = inp.cpu().numpy().reshape(-1) 

            
        #         formatted_vals = [
        #             ("{:.6f}".format(x)).rstrip('0').rstrip('.') 
        #             for x in arr
        #         ]

        #         values_str = ", ".join(formatted_vals)

        #         f.write(f"Element {i} = [ {values_str} ]\n")
        
        scalars = torch.zeros((inputs.shape[0], 1))
        embedded_outputs, _ = self.gatr(embedded_inputs, scalars=scalars)
        
        output = embedded_outputs[:, 0, :]
        x_cluster_coord = self.clustering(output)
        beta = self.beta(output)
        
        x = torch.cat((x_cluster_coord, beta), dim=1)

        # with open("output_dump.txt", "w") as f:
        #     for i, inp in enumerate(x):
        #         arr = inp.cpu().numpy().reshape(-1) 

            
        #         formatted_vals = [
        #             ("{:.6f}".format(x)).rstrip('0').rstrip('.') 
        #             for x in arr
        #         ]

        #         values_str = ", ".join(formatted_vals)

        #         f.write(f"Element {i} = [ {values_str} ]\n")
        return x

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]

        pos_hits_xyz = batch_g.ndata["pos_hits_xyz"]
        hit_type = batch_g.ndata["hit_type"].view(-1, 1)
        vector = batch_g.ndata["vector"]
        input_ = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)
        model_output = self(input_)

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
            tracking=True,
        )
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss)

        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs = []
        y = batch[1]

        batch_g = batch[0]

        pos_hits_xyz = batch_g.ndata["pos_hits_xyz"]
        hit_type = batch_g.ndata["hit_type"].view(-1, 1)
        vector = batch_g.ndata["vector"]
        input_ = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)
        

        with open("inputs_dump.txt", "w") as f:
            for i, inp in enumerate(input_):
                arr = inp.cpu().numpy().reshape(-1) 

            
                formatted_vals = [
                    ("{:.6f}".format(x)).rstrip('0').rstrip('.') 
                    for x in arr
                ]

                values_str = ", ".join(formatted_vals)

                f.write(f"Element {i} = [ {values_str} ]\n")
        
        model_output = self(input_)
        dic = {}
        batch_g.ndata["model_output"] = model_output
        dic["graph"] = batch_g
        dic["part_true"] = y
        
        with open("output_dump.txt", "w") as f:
            for i, inp in enumerate(model_output):
                arr = inp.cpu().numpy().reshape(-1) 

            
                formatted_vals = [
                    ("{:.6f}".format(x)).rstrip('0').rstrip('.') 
                    for x in arr
                ]

                values_str = ", ".join(formatted_vals)

                f.write(f"Element {i} = [ {values_str} ]\n")

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
            tracking=True,
        )
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss, val=True)
        
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
            )
            
            if self.args.predict:
                if len(df_batch) > 0:
                    self.df_showers.append(df_batch)
                    
                if len(df_hits) > 0:
                    self.df_showers_hits.append(df_hits)

        return loss

    def test_step(self, batch, batch_idx):
        self.validation_step_outputs = []
        y = batch[1]

        batch_g = batch[0]

        pos_hits_xyz = batch_g.ndata["pos_hits_xyz"]
        hit_type = batch_g.ndata["hit_type"].view(-1, 1)
        vector = batch_g.ndata["vector"]
        input_ = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)
        
        with open("inputs_dump.txt", "w") as f:
            for i, inp in enumerate(input_):
                arr = inp.cpu().numpy().reshape(-1) 

            
                formatted_vals = [
                    ("{:.6f}".format(x)).rstrip('0').rstrip('.') 
                    for x in arr
                ]

                values_str = ", ".join(formatted_vals)

                f.write(f"Element {i} = [ {values_str} ]\n")
        
        model_output = self(input_)
        dic = {}
        batch_g.ndata["model_output"] = model_output
        dic["graph"] = batch_g
        dic["part_true"] = y
        
        with open("output_dump.txt", "w") as f:
            for i, inp in enumerate(model_output):
                arr = inp.cpu().numpy().reshape(-1) 

            
                formatted_vals = [
                    ("{:.6f}".format(x)).rstrip('0').rstrip('.') 
                    for x in arr
                ]

                values_str = ", ".join(formatted_vals)

                f.write(f"Element {i} = [ {values_str} ]\n")

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
            tracking=True,
        )
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss, val=True)
        
        return loss

    def on_validation_epoch_start(self):
        self.make_mom_zero()
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