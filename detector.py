import cv2
import numpy as np

class ImageDetector:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bf_sift = cv2.BFMatcher()
        self.bf_surf = cv2.BFMatcher(cv2.NORM_L2)
    def compare_image_ORB(self, image1, image2):
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        image1_keypoints, image1_descriptor = self.orb.detectAndCompute(image1_gray, None)
        image2_keypoints, image2_descriptor = self.orb.detectAndCompute(image2_gray, None)
        matches = self.bf_orb.match(image1_descriptor, image2_descriptor)
        matches = sorted(matches, key=lambda x: x.distance)
        result = cv2.drawMatches(image1, image1_keypoints,
                                 image2, image2_keypoints,
                                 matches[100:], None,
                                 flags=2)
        count_of_matches = len(matches)
        return result, count_of_matches
    def comape_image_SIFT(self, image1, image2):
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        image1_keypoints, image1_descriptor = self.sift.detectAndCompute(image1_gray, None)
        image2_keypoints, image2_descriptor = self.sift.detectAndCompute(image2_gray, None)
        matches = self.bf_sift.knnMatch(image1_descriptor, image2_descriptor, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        result = cv2.drawMatchesKnn(image1, image1_keypoints,
                                    image2, image2_keypoints,
                                    good[10:], None,
                                    flags=2)
        count_of_matches = \
            len(good)
        return result, count_of_matches
    def compare_image_FLANN(self, image1, image2):
        kp1, des1 = self.sift.detectAndCompute(image1, None)
        kp2, des2 = self.sift.detectAndCompute(image2, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        result = cv2.drawMatchesKnn(image1, kp1, image2, kp2, matches, None, **draw_params)
        return result