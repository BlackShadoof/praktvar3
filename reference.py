import cv2
import os
from system_settings import *

reference_images = dict()
for root, dirs, files in os.walk(reference_dir_path):
    for file in files:
        path_name = reference_dir_path + '/' + file
        image = cv2.imread(path_name, flags=cv2.IMREAD_COLOR)[...,::-1]
        image = cv2.resize(image, SIZE_IMAGES, interpolation=cv2.INTER_AREA)
        reference_images[file] = image