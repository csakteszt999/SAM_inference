import importlib
import os
import glob
import json
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from pycocotools.coco import COCO
import pandas as pd
from datetime import datetime


## ------------ Python object reader for yaml files -------------------
def load_obj(obj_path:str, default_obj_path:str = "") -> Any:
    """
    Extract python objects from a given path.
    ref: https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given permission.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object '{obj_name}' cannot be loaded from '{obj_path}'.")
    return getattr(module_obj, obj_name)


## -------------- Reading json annotations -----------------------
def get_boxes_from_json(file, desired_id):
    """ Return the boundary boxes and labels for given image_id """
    with open(file, "r") as j:
        data = json.load(j)
    ## 1 ann elem looks like:
    ## {"id":10,"image_id":0,"category_id":2,"bbox":[344,232,25.5,36],"area":918,"segmentation":[],"iscrowd":0},

    bboxes = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        bbox = ann["bbox"]

        if image_id not in bboxes:
            bboxes[image_id] = []
        bboxes[image_id].append({"bbox": bbox})

    boxes_for_desired_id = [bbox["bbox"] for image_id, boxes_list in bboxes.items() if image_id == desired_id for bbox in boxes_list]
    desired_id_boxes = [np.array([[bbox[0], bbox[1]], [bbox[0]+bbox[2], bbox[1]+bbox[3]]]) for bbox in boxes_for_desired_id] # format we need: np.array([[x,y],[width, height]])
    labels_for_desired_id = [np.array([1, i]) for i in range(len(boxes_for_desired_id))]
    # print(f"Converted boxes for id {desired_id}: {desired_id_boxes}")
    return np.array(desired_id_boxes), np.array(labels_for_desired_id)


def get_img_id(file, image_name):
    """ Finding the image id for given image name. """
    with open(file, "r") as j:
        data = json.load(j)

    image_data_dict = {}

    for img_info in data["images"]:
        image_id = img_info["id"]
        file_name = img_info["file_name"]

        if image_id not in image_data_dict:
            image_data_dict[image_id] = file_name

    correct_id = [id for id, name in image_data_dict.items() if image_name == name][0]
    return correct_id

def get_new(jsonfile):
    """ Not finished """
    with open(jsonfile, "r") as j:
        data = json.load(j)

    # Create a dictionary to store image information
    image_data_dict = {}

    # Process annotations to associate bounding boxes with image IDs
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        bbox = np.array([[ann["bbox"][0], ann["bbox"][2]], [ann["bbox"][1], ann["bbox"][3]]])

        if image_id not in image_data_dict:
            image_data_dict[image_id] = {"file_name": None, "boxes": []}

        image_data_dict[image_id]["boxes"].append(bbox)

    # Process images to associate file names with image IDs
    for img_info in data["images"]:
        image_id = img_info["id"]
        file_name = img_info["file_name"]

        if image_id in image_data_dict:
            image_data_dict[image_id]["file_name"] = file_name

    for image_id, image_data in image_data_dict.items():
        print(f"Image ID: {image_id}, File Name: {image_data['file_name']}, Boxes: {image_data['boxes']}")

    return None


#### ------- Pandas dataloading --------
def map_annotation_data_to_df(annotation_path, img_dir_path):
    """ Load json annotation data to a dataframe. Originally made for EfficientSAM. """
    coco = COCO(annotation_path)
    image_ids = coco.getImgIds()
    images_info = coco.loadImgs(image_ids)

    file_map = {}
    bbox_map = {}

    for image_info in images_info:
        file_name = image_info['file_name']
        image_id = image_info['id']

        file_map[file_name] = os.path.join(img_dir_path, file_name)

        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)

        bboxes = []
        for ann in anns:
            bboxes.append(ann['bbox'])
        bbox_map[image_id] = bboxes

    df = pd.DataFrame({'filename': list(file_map.keys()), 'imgpath': list(file_map.values())})
    df['bboxes'] = df.index.map(lambda x: bbox_map[x] if x in bbox_map else None)
    df["input_points"] = df["bboxes"].map(lambda boxes: [np.array([[b[0], b[1]], [b[0] + b[2], b[1] + b[3]]]) for b in boxes])
    df["input_labels"] = df["bboxes"].map(lambda boxes: [np.array([1, i]) for i in range(len(boxes))])
    df["fastsam_points"] = df["bboxes"].map(lambda boxes: [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes])
    return df



## -------- EfficientSAM Visualization functions -------------
## Source: https://github.com/yformer/EfficientSAM

def run_ours_box_or_points(img_path, pts_sampled, pts_labels, model, device):
    image_np = np.array(Image.open(img_path))
    img_tensor = ToTensor()(image_np)
    pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
    pts_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1])
    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].to(device),
        pts_sampled.to(device),
        pts_labels.to(device),
    )

    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )

    return torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="yellow", facecolor=(0, 0, 0, 0), lw=5)
    )

def show_anns_ours(mask, ax):
    ax.set_autoscale_on(False)
    img = np.ones((mask.shape[0], mask.shape[1], 4))
    img[:, :, 3] = 0
    color_mask = [0, 1, 0, 0.7]
    img[np.logical_not(mask)] = color_mask
    ax.imshow(img)

def custom_pred_show(original_image, mask, bboxes, ax):
    ax.imshow(original_image)
    ax.set_autoscale_on(False)
    masked_img = np.ones((mask.shape[0], mask.shape[1], 4))
    masked_img[:, :, 3] = 0
    color_mask = [0, 1, 0, 0.7]
    masked_img[np.logical_not(mask)] = color_mask
    ax.imshow(masked_img)
    for b in bboxes:
        x1, y1, x2, y2 = b[0][0], b[0][1], b[1][0], b[1][1]
        show_box(box=[x1, y1, x2, y2], ax=ax)



## -------- Batch processing functions ---------

## Batch processing on FastSAM
from apps.FastSAM.fastsam.prompt import FastSAMPrompt
def batch_process_fast(input_folder, output_folder, annotation_file, model, device, retina_masks, imgsz, conf, iou):
    model.to(device)
    inp = os.path.abspath(input_folder)
    out = os.path.abspath(output_folder)
    dataframe = map_annotation_data_to_df(annotation_path=annotation_file, img_dir_path=inp)

    start = datetime.now()
    print("Start batch processing.")
    for i in range(len(dataframe)):
        image, im_path, bboxes, fsam_boxes = dataframe.iloc[i, 0], dataframe.iloc[i, 1], dataframe.iloc[i, 2], dataframe.iloc[i, 5]

        everything_results = model(source=os.path.join(inp, image),
                                   device=device,
                                   retina_masks=retina_masks,
                                   imgsz=imgsz,
                                   conf=conf,
                                   iou=iou)

        prompt_process = FastSAMPrompt(os.path.join(inp, image), everything_results, device=device)
        ann = prompt_process.box_prompt(bboxes=fsam_boxes)
        output = os.path.join(out, f"fastsam_img_{image.split('_')[0]}.png")
        prompt_process.plot(annotations=ann, output_path=output, bboxes=fsam_boxes)

    time_elapsed = datetime.now() - start
    print(f"Batch processing time (hh:mm:ss:ms): {time_elapsed}")


## Batch processing on EfficientSAM
def batch_process_effsam(input_folder, output_folder, annotation_file, device):
    from apps.EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt
    model = build_efficient_sam_vitt()
    model.to(device)
    model.eval()
    inp = os.path.abspath(input_folder)
    out = os.path.abspath(output_folder)

    dataframe = map_annotation_data_to_df(annotation_path=annotation_file, img_dir_path=inp)
    # print(f"Filenames:\n{dataframe['filename']}\n")
    # print(f"Boxes:\n{dataframe['bboxes']}\n")
    # print(f"Input_points:\n{dataframe['input_points']}\n")
    # print(f"Input_labels:\n{dataframe['input_labels']}\n")

    start = datetime.now()
    print("Start batch processing.")
    for i in range(len(dataframe)):
        image, im_path, bboxes, inp_pts, inp_lbls = dataframe.iloc[i, 0], dataframe.iloc[i, 1], dataframe.iloc[i, 2], dataframe.iloc[i, 3], dataframe.iloc[i, 4]

        sample_image_np = np.array(Image.open(os.path.join(inp, image)))

        inp_pts, inp_lbls = np.array(inp_pts), np.array(inp_lbls)
        mask = run_ours_box_or_points(img_path=os.path.join(inp, image),
                                             pts_sampled=inp_pts,
                                             pts_labels=inp_lbls,
                                             model=model,
                                             device=device,
                                      )

        fname = os.path.join(out, f"effsam_img_{image.split('_')[0]}.png")
        fig, ax = plt.subplots(1,1, figsize=(14,12))
        custom_pred_show(original_image=sample_image_np, mask=np.array(mask), bboxes=inp_pts, ax=ax)
        plt.axis("off")
        plt.savefig(fname=fname, format="png")

    time_elapsed = datetime.now() - start
    print(f"Batch processing time (hh:mm:ss:ms): {time_elapsed}")



