import os
import numpy as np
import shutil

data_path = '../data/'
data_name = 'UAVid'
ann_dir = 'ann_dir/test'
img_dir = 'img_dir/test'
distill_num = 50

origin_data_path = os.path.join(data_path, data_name)
distill_data_path = os.path.join(data_path, data_name + '_distill')
origin_ann_dir = os.path.join(origin_data_path, ann_dir)
origin_img_dir = os.path.join(origin_data_path, img_dir)
distill_ann_dir = os.path.join(distill_data_path, ann_dir)
distill_img_dir = os.path.join(distill_data_path, img_dir)

os.makedirs(distill_data_path)
os.makedirs(distill_ann_dir)
os.makedirs(distill_img_dir)

imgs = os.listdir(origin_img_dir)
data_nums = len(imgs)
print(data_name + ' have ' + str(data_nums) + ' images')
samples = np.random.choice(imgs, size=distill_num, replace=False)
for sample in samples:
    # ann_name = sample.split('.')[0] + '_instance_color_RGB.png'
    ann_name = sample
    shutil.copy(os.path.join(origin_ann_dir, ann_name), os.path.join(distill_ann_dir, ann_name))
    shutil.copy(os.path.join(origin_img_dir, sample), os.path.join(distill_img_dir, sample))