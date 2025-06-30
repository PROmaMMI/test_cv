import os
import cv2
import albumentations as A
import numpy as np

train_images_dir = 'dataset/images/train'
train_labels_dir = 'dataset/labels/train'

def read_yolo_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        bboxes.append([x_center, y_center, width, height, class_id])
    return bboxes

def write_yolo_label(label_path, bboxes):
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            x_center, y_center, width, height, class_id = bbox
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


for image_file in os.listdir(train_images_dir):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(train_images_dir, image_file)
        label_path = os.path.join(train_labels_dir, image_file.replace('.jpg', '.txt'))


        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        if not os.path.exists(label_path):
            continue  
        bboxes = read_yolo_label(label_path)
        if not bboxes:
            continue  

   
        bboxes_yolo = [bbox[:4] for bbox in bboxes]
        class_labels = [bbox[4] for bbox in bboxes]

   
        augmented = augmentations(image=image, bboxes=bboxes_yolo, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_class_labels = augmented['class_labels']

    
        aug_image_file = f"aug_{image_file}"
        aug_image_path = os.path.join(train_images_dir, aug_image_file)
        cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

  
        aug_label_path = os.path.join(train_labels_dir, aug_image_file.replace('.jpg', '.txt'))
        aug_bboxes_with_class = [[*bbox, class_id] for bbox, class_id in zip(aug_bboxes, aug_class_labels)]
        write_yolo_label(aug_label_path, aug_bboxes_with_class)

print("Аугментация тренировочной выборки завершена!")