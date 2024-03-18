import os
import torch
import torch.nn
from omegaconf import DictConfig
import hydra

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modules folder to collect custom python codes
from modules.utils import load_obj
from modules.utils import batch_process_fast, batch_process_effsam
from modules import mobsam_inf

@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def my_app(cfg: DictConfig) -> None:

    if cfg.sam_app.model_name == "FastSAM_batch":
        print("Running inference on FastSAM...")
        model = load_obj(cfg.sam_app.model)(model=cfg.paths.fast_sam_path)

        batch_process_fast(
            model=model,
            input_folder=cfg.paths.img_path,
            output_folder=cfg.paths.out_path,
            annotation_file=cfg.paths.annot_path,
            device=device,
            retina_masks=cfg.hparams.retina_masks,
            imgsz=cfg.hparams.img_size,
            conf=cfg.hparams.conf,
            iou=cfg.hparams.iou,
        )

    if cfg.sam_app.model_name == "MobileSAMv2_batch":
        print("Running inference on MobileSAMv2...")
        out_dir = cfg.paths.out_path
        img_dir = cfg.paths.img_path
        mobsam_dir = cfg.paths.mobsam_dir
        ann_file = cfg.paths.annot_path
        mobsam_inf.run(output_dir=out_dir, input_dir=img_dir, mobsam_dir=mobsam_dir, annotation_file=ann_file, encoder_type="sam_vit_h")

    if cfg.sam_app.model_name == "EfficientSAM_batch":
        print("Running inference on EfficientSAM...")
        batch_process_effsam(input_folder=cfg.paths.img_path,
                             output_folder=cfg.paths.out_path,
                             annotation_file=cfg.paths.annot_path,
                             device=device,
        )


if __name__ == "__main__":
    my_app()

