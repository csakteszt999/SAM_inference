from pycocotools.coco import COCO
import os
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

def map_annotation_data_to_df(annotation_path, img_dir_path):
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
    return df

# im_dir = "./../imgs/wp_detection/test"
# ann_file = "./../imgs/wp_detection/test/_annotations.coco.json"
# df, file_map, bbox_map = create_names_path_bbox_dict(annotation_path=ann_file, img_dir_path=im_dir)
# print(df)

def get_data_by_filename(df, filename):
    df = df.copy()
    data = df[df["filename"] == filename]
    if not data.empty:
        image_id = data.index[0]
        name = data["filename"].iloc[0]
        bboxes = data["bboxes"].iloc[0]
    else:
        print("Filename not found in the DataFrame. ")
    return image_id, name, bboxes

# file_name = "011713_png.rf.81bfe586e17ce476854b119ce37cbe7c.jpg"
# id, name, bboxes = get_data_by_filename(df=df, filename=file_name)
# print(f"Id: {id}, Name: {name}, Boxes: {bboxes}")


def create_annotation_datastructure_for_training(annotation_file_paths):
    category_names = {}
    file_names_with_bboxes = []
    for train_i in annotation_file_paths:
        coco = COCO(train_i)  # Replace with your COCO JSON file path
        # Get all image IDs and their respective file names
        image_info = coco.loadImgs(coco.getImgIds())
        image_id_to_file_name = {image['id']: image['file_name'] for image in image_info}
        # Get all annotations and connect them with the corresponding file names
        annotations = coco.loadAnns(coco.getAnnIds())

        # Load all categories
        categories = coco.loadCats(coco.getCatIds())
        # Extract category names
        for category in categories:
            category_names[category['id']] = category['name']
        # List file names with connected bounding boxes
        for annotation in annotations:
            image_id = annotation['image_id']
            file_name = image_id_to_file_name.get(image_id)
            if file_name:
                bbox = annotation['bbox']  # list(map(int, annotation['bbox']))
                catid = annotation['category_id']
                file_names_with_bboxes.append({'file_name': file_name, 'bbox': bbox, 'labels': category_names[catid]})
                # print(bbox)

    df_annot = pd.DataFrame(file_names_with_bboxes)
    return category_names, df_annot

# cat_names, df_annot = create_annotation_datastructure_for_training(annotation_file_paths=[ann_file])
# print(cat_names)
# print(df_annot)



