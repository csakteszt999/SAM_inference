import torch
import cv2
import os
from datetime import datetime
from typing import Any, Dict, Generator, List
import matplotlib.pyplot as plt
import numpy as np

from apps.MobileSAM.MobileSAMv2.mobilesamv2 import sam_model_registry, SamPredictor
from .utils import map_annotation_data_to_df, show_box

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def create_model(mobsam_dir):
    Prompt_guided_path = f'{mobsam_dir}/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt'
    sam_vit_h_path = f'{mobsam_dir}/weight/sam_vit_h.pt'

    PromptGuidedDecoder = sam_model_registry['PromptGuidedDecoder'](Prompt_guided_path)
    mobilesamv2 = sam_model_registry['vit_h'](checkpoint=sam_vit_h_path) ## Image encoder
    mobilesamv2.prompt_encoder = PromptGuidedDecoder['PromtEncoder']
    mobilesamv2.mask_decoder = PromptGuidedDecoder['MaskDecoder']
    return mobilesamv2

def show_anns(anns):
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:, :, 3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m = m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    ax.imshow(img)

def show_anns_mod(original_image, anns, bboxes, ax):
    if len(anns) == 0:
        return
    ax.imshow(original_image)
    ax.set_autoscale_on(False)
    masked_img = np.ones((anns.shape[1], anns.shape[2], 4))
    masked_img[:, :, 3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m = m.cpu().numpy()
        # color_mask = np.concatenate([np.random.random(3), [1]])
        color_mask = [0, 1, 0, 0.7]
        masked_img[m] = color_mask
    ax.imshow(masked_img)
    for b in bboxes:
        x1, y1, x2, y2 = b[0][0], b[0][1], b[1][0], b[1][1]
        show_box(box=[x1, y1, x2, y2], ax=ax)

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size: (b + 1) * batch_size] for arg in args]

def run(output_dir, input_dir, mobsam_dir, annotation_file, encoder_type):

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    mob_abs = os.path.abspath(mobsam_dir)

    encoder_path = {'efficientvit_l2': './weight/l2.pt',
                    'tiny_vit': f'{mob_abs}/MobileSAMv2/weight/mobile_sam.pt',
                    'sam_vit_h': f'{mob_abs}/weight/sam_vit_h.pt', }

    mobilesamv2 = create_model(mobsam_dir=mob_abs)
    image_encoder = sam_model_registry[encoder_type](encoder_path[encoder_type])
    mobilesamv2.image_encoder = image_encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobilesamv2.to(device=device)
    mobilesamv2.eval()
    predictor = SamPredictor(mobilesamv2)

    dataframe = map_annotation_data_to_df(annotation_path=annotation_file, img_dir_path=input_dir)

    start = datetime.now()
    print("Start batch processing.")
    for i in range(len(dataframe)):
        image_name, image_path, inp_points, fsam_boxes = dataframe.iloc[i, 0], dataframe.iloc[i, 1], dataframe.iloc[i, 3], dataframe.iloc[i, 5]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        input_boxes = np.array(fsam_boxes) # Bx4
        input_boxes = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes).cuda()

        sam_mask = []
        image_embedding = predictor.features
        image_embedding = torch.repeat_interleave(image_embedding, 320, dim=0)
        prompt_embedding = mobilesamv2.prompt_encoder.get_dense_pe()
        prompt_embedding = torch.repeat_interleave(prompt_embedding, 320, dim=0)

        for (boxes,) in batch_iterator(320, input_boxes):
            with torch.no_grad():
                image_embedding = image_embedding[0:boxes.shape[0], :, :, :]
                prompt_embedding = prompt_embedding[0:boxes.shape[0], :, :, :]
                sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None, )
                low_res_masks, _ = mobilesamv2.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=prompt_embedding,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )
                low_res_masks = predictor.model.postprocess_masks(low_res_masks, predictor.input_size,
                                                                  predictor.original_size)
                sam_mask_pre = (low_res_masks > mobilesamv2.mask_threshold) * 1.0
                sam_mask.append(sam_mask_pre.squeeze(1))
        sam_mask = torch.cat(sam_mask)
        annotation = sam_mask
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=True)
        show_img = annotation[sorted_indices]

        fig, ax = plt.subplots(1,1, figsize=(14,12))
        show_anns_mod(original_image=image, anns=show_img, bboxes=inp_points, ax=ax)
        plt.axis('off')
        fname = os.path.join(output_dir, f"mobilesam_img_{image_name.split('_')[0]}.png")
        plt.savefig(fname=fname, format="png", bbox_inches='tight', pad_inches=0.0)

    time_elapsed = datetime.now() - start
    print(f"Batch processing time (hh:mm:ss:ms): {time_elapsed}")
