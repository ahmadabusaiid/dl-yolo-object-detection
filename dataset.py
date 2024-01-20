import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None, augment=False, aug_prob = 0.3
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.augment = augment
        self.aug_prob = aug_prob
        if self.augment:
            self.affine_aug = A.Affine(translate_percent=0.2, p=0.2)
            self.h_flip_aug = A.HorizontalFlip(p=0.2)
            self.rand_bright_ctr_aug = A.RandomBrightnessContrast(p=0.2)
            self.rotate_aug = A.Rotate(limit=30, p=0.2)
            self.scale_aug = A.Affine(scale=(0.6, 0.8), p=0.2, keep_ratio=True)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if ((float(x) != int(float(x)))) else int(float(x))
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
        
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        boxes = torch.tensor(boxes)

        if self.augment:
            rand_val = random.random()
            if rand_val <= self.aug_prob:
                boxes_only = boxes[:, 1:].tolist()
                labels_only = list(map(lambda sublist: list(map(str, sublist)), boxes[:, :1].tolist()))
                boxes_alb_yolo_format = [box + labels_only[i] for i, box in enumerate(boxes_only)]

                augmentations = A.Compose([self.affine_aug, self.h_flip_aug, self.rand_bright_ctr_aug, 
                                        self.rotate_aug, self.scale_aug], bbox_params=A.BboxParams(format='yolo'))

                augmented_data = augmentations(image=np.array(image), bboxes=boxes_alb_yolo_format)
                image_aug = augmented_data['image']
                boxes_alb_yolo_format = augmented_data['bboxes']

                if len(boxes_alb_yolo_format) > 0:
                    boxes_only = np.array(boxes_alb_yolo_format)[:, :-1].astype(float)
                    labels_only = np.array(boxes_alb_yolo_format)[:, -1:].astype(float)
                    boxes_aug = np.append(labels_only, boxes_only, axis=1)

                    image = Image.fromarray(image_aug)
                    boxes = torch.tensor(boxes_aug)


        if self.transform:
            image, boxes = self.transform(image, boxes)
            
        
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (
                width * self.S,
                height * self.S
            )

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, -9:-5] = box_coordinates
                label_matrix[i, j, class_label] = 1
        
        # return image, label_matrix, label_path
        return image, label_matrix