import os
import random
import shutil
from sklearn.model_selection import train_test_split


frames_dir = 'data/frames'  
labels_dir = 'data/labels'  
dataset_dir = 'dataset'     


images = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]


train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)


def copy_files(file_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    for file in file_list:
        shutil.copy(os.path.join(src_img_dir, file), dst_img_dir)
        label_file = file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(src_lbl_dir, label_file)):
            shutil.copy(os.path.join(src_lbl_dir, label_file), dst_lbl_dir)
        else:
            print(f"Предупреждение: аннотация для {file} не найдена!")

copy_files(train_images, frames_dir, labels_dir, 
           os.path.join(dataset_dir, 'images', 'train'), 
           os.path.join(dataset_dir, 'labels', 'train'))
copy_files(val_images, frames_dir, labels_dir, 
           os.path.join(dataset_dir, 'images', 'val'), 
           os.path.join(dataset_dir, 'labels', 'val'))
copy_files(test_images, frames_dir, labels_dir, 
           os.path.join(dataset_dir, 'images', 'test'), 
           os.path.join(dataset_dir, 'labels', 'test'))

print("Разбиение завершено!")