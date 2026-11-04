
import numpy as np
import numpy as np
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def clearfolder(train_path):
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    else:
        for i in os.listdir(train_path):
            os.remove(os.path.join(train_path,i))


overlap=0
split_width=1024
split_height=1024
count = 0

data_path = 'C:/dataset/GDCLD_ori/'
imgs_path = data_path+'test_data/'
labels_path = data_path+'test_label/'
out_path = 'C:/dataset/GDCLD/'
outputpath1 = out_path+'img_dir/'
outputpath2 = out_path+'ann_dir/'
clearfolder(outputpath1)
clearfolder(outputpath2)

for area in os.listdir(imgs_path):
    area_path = os.path.join(imgs_path, area)
    for img in os.listdir(area_path):
        img1 = os.path.join(area_path, img)
        label = img1.replace('test_data', 'test_label')
        img1= np.asarray(Image.open(img1))
        label = np.asarray(Image.open(label).convert('L')).copy()
        label[label > 0] = 1
        img_h1, img_w1,_ = img1.shape
        img_h2, img_w2 = label.shape
        img_h = min(img_h1, img_h2)
        img_w = min(img_w1, img_w2)
        X_points = start_points(img_w, split_width, overlap)
        Y_points = start_points(img_h, split_height, overlap)
        for i in Y_points:
            for j in X_points:
                split1 = img1[i:i + split_height, j:j + split_width,:]
                split2 = label[i:i + split_height, j:j + split_width]
                img_h1, img_w1, _ = split1.shape
                img_h2, img_w2 = split2.shape
                if img_h1 != img_h2 or img_w1 != img_w2:
                    print('error for image'+str(count))
                    continue
                Image.fromarray(split1).save(os.path.join(outputpath1, str(count) + ".png"))
                Image.fromarray(split2).save(os.path.join(outputpath2, str(count) + ".png"))
                count += 1
