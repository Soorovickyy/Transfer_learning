import torch
import os

data_path = 'flower_photos'
base_path = 'dataset'

val_split = 0.1
train = os.path.join(base_path, 'train')
val = os.path.join(base_path, 'val')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_size = 224

#device = "cuda" if torch.cuda.is_available() else "cpu"

feature_extraction_batch_size = 256
pred_batch_size = 4
epoch = 2
lr = 0.001

