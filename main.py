import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
from omegaconf import DictConfig, OmegaConf
import hydra
from PIL import Image
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modules folder to collect python codes
from modules.log_types import CustomLogger
from modules.utils import load_obj, show_points, show_anns_ours, show_box, run_ours_box_or_points
from modules import mobsam_inf

@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def my_app(cfg: DictConfig) -> None:

    if cfg.sam_app.model_name == "FastSAM":
        print("Running inference on FastSAM...")
        model = load_obj(cfg.sam_app.model)(model=cfg.paths.fast_sam_path)
        everything_results = model(cfg.paths.img_path,
                                   device=device,
                                   retina_masks=cfg.hparams.retina_masks,
                                   imgsz=cfg.hparams.img_size,
                                   conf=cfg.hparams.conf,
                                   iou=cfg.hparams.iou)
        prompt_process = load_obj(cfg.sam_app.prompt)(cfg.paths.img_path, everything_results, device=device)
        obj_boxes = [[310,110,390,190],[410,140,490,220]]
        ann = prompt_process.box_prompt(bboxes=obj_boxes)
        # ann = prompt_process.everything_prompt()
        # ann = prompt_process.text_prompt(text="humans in da water")

        out = cfg.paths.out_path + f"/fastsam_{1}.jpg"
        prompt_process.plot(annotations=ann, output_path=out, bboxes=obj_boxes)

    if cfg.sam_app.model_name == "MobileSAM":
        print("Running inference on MobileSAM...")
        # model = load_obj(cfg.sam_app.model)(checkpoint=cfg.paths.mobile_sam_path) # Not a class
        from apps.MobileSAM.mobile_sam import sam_model_registry
        model_type = "vit_t"
        model = sam_model_registry[model_type](checkpoint=cfg.paths.mobile_sam_path)
        model.to(device=device)
        model.eval()
        predictor = load_obj(cfg.sam_app.predictor)(sam_model=model)

        # Get image preprocess
        img_arr = np.asarray(Image.open(cfg.paths.img_path), dtype=np.uint8)
        predictor.set_image(img_arr)
        obj_boxes = np.array(OmegaConf.to_object(cfg.hparams.box_prompt))
        # masks, iou_preds, lowres_masks = predictor.predict(box=obj_boxes)
        masks, iou_preds, lowres_masks = predictor.predict()
        print(masks[:3, :3, :3], iou_preds)
        ## -> No bbox output, but we can calcualte it with x-y min max values of mask

    if cfg.sam_app.model_name == "MobileSAMv2":
        print("Running inference on MobileSAMv2...")

        out_dir = "./preds"
        img_path = "./imgs/"
        mobsam_inf.run(output_dir=out_dir, img_path=img_path, encoder_type="sam_vit_h") # sam_vit_h or tiny_vit
        # Error: No output bbox with YOLO encoder

    if cfg.sam_app.model_name == "EfficientSAM":
        print("Running inference on EfficientSAM...")
        from apps.EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt

        model = build_efficient_sam_vitt()
        # pred_logits, pred_iou = model()

        obj_boxes = np.array(OmegaConf.to_object(cfg.hparams.box_prompt))
        new_box = np.array([obj_boxes[:2],obj_boxes[2:]])
        input_label = np.array([2, 3])

        fig, ax = plt.subplots(1,2, figsize=(12,12))
        img_dir = os.path.abspath(cfg.paths.img_path)
        img = os.listdir(img_dir)[0]
        img_arr = np.asarray(Image.open(os.path.join(img_dir, img)), dtype=np.uint8)

        ## show input img and boxes
        show_points(new_box, input_label, ax[0])
        show_box(obj_boxes, ax[0])
        ax[0].imshow(img_arr)

        ## show mask prediction
        ax[1].imshow(img_arr)
        mask_out = run_ours_box_or_points(img_path=os.path.join(img_dir, img),
                                          pts_sampled=new_box,
                                          pts_labels=input_label,
                                          model=model)
        show_anns_ours(mask_out, ax[1])
        ax[1].title.set_text("EfficientSAM (VIT-tiny)")
        ax[1].axis('off')
        plt.show()



if __name__ == "__main__":
    my_app()

