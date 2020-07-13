import matplotlib.pyplot as plt
import cv2
import random
import os
from detector import ImageDetector
from reference import *
from system_settings import *


if __name__ == '__main__':
    detector = ImageDetector()
    houses = list()
    for root, dirs, files in os.walk(house_dir_path):
        for file in files:
            houses.append(file)
    path_name = house_dir_path + '/' + random.choice(houses)
    print('Selected file ' + path_name)
    image1 = cv2.imread(path_name, flags=cv2.IMREAD_COLOR)[...,::-1]
    image1 = cv2.resize(image1, SIZE_IMAGES, interpolation=cv2.INTER_AREA)

    count_matches_dict = dict()
    for image_key, image in reference_images.items():
        result_image_SIFT, count_SIFT = detector.comape_image_SIFT(image1, image)
        count_matches_dict[image_key] = count_SIFT
    if max(count_matches_dict.values()) < min_count_matches:
        print('No matches found')
        exit()
    range_image = sorted(count_matches_dict, key=count_matches_dict.get, reverse=True)
    final_path = reference_dir_path + '/' + range_image[0]
    final_image = cv2.imread(final_path, flags=cv2.IMREAD_COLOR)[..., ::-1]
    final_image = cv2.resize(final_image, SIZE_IMAGES, interpolation=cv2.INTER_AREA)
    result_image_FLANN = detector.compare_image_FLANN(image1, final_image)
    result_image_SIFT, count_SIFT = detector.comape_image_SIFT(image1, final_image)
    fig, axes = plt.subplots(1, 2, figsize=(20, 18))
    plt.title('House Detection')
    axes[0].set_title('Best Matching Points FLANN')
    axes[0].imshow(result_image_FLANN)
    axes[1].set_title('Best Matching Points SIFT')
    axes[1].imshow(result_image_SIFT)
    plt.show()
    print("\nNumber of Matching Keypoints SIFT: ", count_SIFT)