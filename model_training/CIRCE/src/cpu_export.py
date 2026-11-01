"""Save a ready-to-load CPU graph alongside each validation checkpoint."""

from pathlib import Path
import warnings

from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from src.cgatr.cpu_inference import export_cpu_checkpoint


class CPUExportModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, cpu_export_library=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_export_library = cpu_export_library
        self._cpu_example = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (not self.cpu_export_library or self._cpu_example is not None
                or not trainer.is_global_zero or trainer.sanity_checking or batch is None):
            return
        features = batch["features"]
        if features.ndim != 2 or features.shape[1] != 10:
            return
        # A few measured hits cover both embedding paths; no truth labels are used.
        vertex = features[features[:, 3] == 0][:4]
        drift = features[features[:, 3] == 1][:4]
        if len(vertex) and len(drift):
            self._cpu_example = torch.cat([vertex, drift]).detach().cpu().clone()

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if not trainer.is_global_zero or not self.cpu_export_library:
            return
        try:
            if self._cpu_example is None:
                raise ValueError("No validation example containing both hit types")
            output = Path(filepath).with_suffix(".cpu.pt")
            source = export_cpu_checkpoint(filepath, self._cpu_example, self.cpu_export_library, output)
            print(f"[cgatr_fcc] CPU inference model ({source}): {output}", flush=True)
        except Exception as error:
            # Preserve the resume checkpoint even if the optional native library
            # is incompatible with this training environment.
            warnings.warn(f"Saved {filepath}, but CPU inference export failed: {error}")
