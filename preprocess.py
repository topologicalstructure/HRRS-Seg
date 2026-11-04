import cv2
import numpy as np

def pre_process(img, clipLimit=2.0, tileSize=(8,8), sigma=1.0, alpha=1.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Clahe on L channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileSize)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img_clahe = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Unsharp mask
    blur = cv2.GaussianBlur(img_clahe, (0, 0), sigmaX=sigma)
    img_sharp = cv2.addWeighted(img_clahe, 1 + alpha, blur, -alpha, 0)
    return img_sharp


path = 'demo/AOI_2_Vegas_20__rgb.png'
image_sharp = pre_process(cv2.imread(path))
cv2.imwrite('output.png', image_sharp)