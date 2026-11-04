import cv2
import numpy as np
import os

labels_path = 'D:/git/HRRS-Seg/data/GVLM/ann_dir/test'

labels = os.listdir(labels_path)
for label in labels:
    label_path = os.path.join(labels_path, label)
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    for x in range(label_img.shape[0]):
        for y in range(label_img.shape[1]):
            if label_img[x][y] == 255:
                label_img[x][y] = 1
    cv2.imwrite(label_path, label_img)

