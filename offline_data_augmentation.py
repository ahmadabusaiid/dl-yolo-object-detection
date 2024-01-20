import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from PIL import Image

# get all train image names
# sample x% of the images
# apply stratified augmentation for both image and their corresponding bounding boxes
# save images and bounding boxes in yolo format
# update train.csv

DATA_PATH = 'data'
AUG_FRAC = 0.3
IMG_DIM = 448

affine_aug = A.Affine(translate_percent=0.2, p=1)
h_flip_aug = A.HorizontalFlip(p=1)
rand_bright_ctr_aug = A.RandomBrightnessContrast(p=1)
rotate_aug = A.Rotate(limit=30, p=1)
scale_aug = A.Affine(scale=(0.6, 0.8), p=1, keep_ratio=True)
# crop_aug = A.RandomCrop(p=1, width=300, height=300)

augmentors = [
    A.Compose([affine_aug, h_flip_aug], bbox_params=A.BboxParams(format='yolo')),
    A.Compose([h_flip_aug, rand_bright_ctr_aug], bbox_params=A.BboxParams(format='yolo')),
    A.Compose([rotate_aug, h_flip_aug], bbox_params=A.BboxParams(format='yolo')),
    A.Compose([scale_aug, affine_aug], bbox_params=A.BboxParams(format='yolo')),
    A.Compose([scale_aug, rotate_aug], bbox_params=A.BboxParams(format='yolo')),
]

def extract_files(path, aug_fraction):
    df = pd.read_csv(f"{path}/train.csv", header=None, names=["image", "label"])
    df = df.sample(frac=aug_fraction, random_state=123).reset_index(drop=True)

    images = []
    for file_name in df.image.values.tolist():
        image_path = os.path.join(f"{path}/images/", file_name)
        image = Image.open(image_path)
        image_array = np.array(image)
        images.append(image_array)

    labels_raw = []
    for file_name in df.label.values.tolist():
        label_path = os.path.join(f"{path}/labels/", file_name)
        with open(label_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            labels_raw.append(lines)
        labels = [
            [
                [int(value) if i == 0 else float(value) for i, value in enumerate(item.split())]
                for item in nested_list
            ] 
            for nested_list in labels_raw
        ]

    return df, images, labels

def create_partitions(original_list):
    elements_per_sublist = len(original_list) // 5
    sublists = [original_list[i * elements_per_sublist: (i + 1) * elements_per_sublist] for i in range(5)]
    sublists[-1] += original_list[5 * elements_per_sublist:]
    return sublists



train_df, train_images, train_labels = extract_files(DATA_PATH, AUG_FRAC)
train_images_partitioned = create_partitions(train_images)
train_labels_partitioned = create_partitions(train_labels)

aug_images, aug_labels = [], []
train_labels_for_viz = []
for i, aug in enumerate(augmentors):
    for j, (image, bb) in enumerate(zip(train_images_partitioned[i], train_labels_partitioned[i])):
        bboxes = [box[1:] + [str(box[:1][0])] for box in bb]
        train_labels_for_viz.append(bboxes)
        transformed = aug(image=image, bboxes=bboxes)
        aug_images.append(transformed['image'])
        aug_labels.append(transformed['bboxes'])

for i, image_file in enumerate(train_df.image.values.tolist()):
    cv2.imwrite(f"{DATA_PATH}/aug_images/{image_file[:-4]+'_999999'+image_file[-4:]}", aug_images[i])
    reverted_format_aug_labels = []
    for j, lab in enumerate(aug_labels[i]):
        reverted_format_aug_labels.append(' '.join(map(str, [int(aug_labels[i][j][-1:][0])] + list(aug_labels[i][j][:-1]))))
    with open(f"{DATA_PATH}/aug_labels/{image_file[:-4]+'_999999.txt'}", 'w') as file:
        file.writelines('\n'.join(map(str, reverted_format_aug_labels)))

full_train_df = pd.read_csv(f"{DATA_PATH}/train.csv", header=None, names=["image", "label"])
train_df["image"] = train_df["image"].map(
    lambda x: x[:-4]+'_999999'+x[-4:]
)
train_df["label"] = train_df["label"].map(
    lambda x: x[:-4]+'_999999.txt'
)
train_aug = pd.concat([full_train_df, train_df]).sample(frac=1).reset_index(drop=True)
train_aug.to_csv(f"{DATA_PATH}/train_aug.csv", header=None, index=False)

print(f"\n*** Completed augmentation for {len(train_df)} images ***\n")

