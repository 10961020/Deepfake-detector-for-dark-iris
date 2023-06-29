# import matlab.engine
import numpy as np
import dlib
import cv2
import os
import torch
from skimage import exposure
from skimage.feature import canny
from skimage.morphology import convex_hull_image
from scipy.ndimage.morphology import binary_dilation
from skimage.transform import hough_circle, hough_circle_peaks

import matplotlib.pyplot as plt
from skimage import filters
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# eng = matlab.engine.start_matlab()

LANDMARKS = {"mouth": (48, 68),
             "mouth_inner": (60, 68),
             "right_eyebrow":(17, 22),
             "left_eyebrow": (22, 27),
             "right_eye": (36, 42),
             "left_eye": (42, 48),
             "nose": (27, 35),
             "jaw": (0, 17),
             }

MOUTH_LM = np.arange(LANDMARKS["mouth_inner"][0], LANDMARKS["mouth"][1])
LEYE_LM = np.arange(LANDMARKS["left_eye"][0], LANDMARKS["left_eye"][1])
REYE_LM = np.arange(LANDMARKS["right_eye"][0], LANDMARKS["right_eye"][1])


def shape_to_np(shape):
    number_of_points = shape.num_parts
    points = np.zeros((number_of_points, 2), dtype=np.int32)
    for i in range(0, number_of_points):
        points[i] = (shape.part(i).x, shape.part(i).y)

    return points


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


# TODO 得到眼部区域的mask
def generate_convex_mask(shape, points_x, points_y):
    mask = np.zeros(shape, dtype=np.uint8)

    #clip to image size
    points_x = np.clip(points_x, 0, max(0, shape[1] - 1))
    points_y = np.clip(points_y, 0, max(0, shape[0] - 1))

    #set mask pixels
    mask[points_y, points_x] = 255
    mask = convex_hull_image(mask)

    return mask


# TODO 加载dlib人脸关键点检测
def load_facedetector(config):
    """Loads dlib face and landmark detector."""
    # download if missing http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    if not os.path.isfile(config['facedetector_path']):
        print('Could not find shape_predictor_68_face_landmarks.dat.')
        exit(-1)
    face_detector = dlib.get_frontal_face_detector()
    landmark_Predictor = dlib.shape_predictor(config['facedetector_path'])

    return face_detector, landmark_Predictor


# TODO 裁剪眼睛覆盖区域 320*280xp  返回裁剪后眼部区域以及对应的眼部mask
def get_crops_eye(face_detector, landmark_Predictor, img, input_file):
    faces = face_detector(img, 1)
    img_eye_crop_net = []
    img_eye_mask_net = []
    img_eye_crop = []
    img_eye_mask = []
    for face in faces:
        landmarks = landmark_Predictor(img, face)  # get 68 landmarks for each face
        landmarks_np = shape_to_np(landmarks)
        # for i, j in [(36, 39), (42, 45)]:
        for i in [LEYE_LM, REYE_LM]:
            eye_mark_local = landmarks_np[i]
            eye_mask = generate_convex_mask(img[..., 0].shape, eye_mark_local[..., 0], eye_mark_local[..., 1])
            eye_mask = eye_mask.astype('uint8')

            pt_pos_left, pt_pos_right = landmarks_np[i[0]], landmarks_np[i[3]]
            center_point = ((pt_pos_right[0] - pt_pos_left[0]) // 2 + pt_pos_left[0], (pt_pos_right[1] - pt_pos_left[1]) // 2 + pt_pos_left[1])
            try:
                img_eye_crop_net.append(cv2.resize(img[center_point[1] - 70:center_point[1] + 70, center_point[0] - 80:center_point[0] + 80], (320,280)))   # TODO 缩放影响canny???
                img_eye_mask_net.append(cv2.resize(eye_mask[center_point[1] - 70:center_point[1] + 70, center_point[0] - 80:center_point[0] + 80], (320,280)))
                img_eye_crop.append(img[center_point[1] - 70:center_point[1] + 70, center_point[0] - 80:center_point[0] + 80])
                img_eye_mask.append(eye_mask[center_point[1] - 70:center_point[1] + 70, center_point[0] - 80:center_point[0] + 80])
                # img_eye_crop.append(img[center_point[1] - 30:center_point[1] + 15, center_point[0] - 20:center_point[0] + 24])
                # img_eye_mask.append(eye_mask[center_point[1] - 30:center_point[1] + 15, center_point[0] - 20:center_point[0] + 24])
            except:
                print("123: ", input_file)

    return img_eye_crop_net, img_eye_mask_net, img_eye_crop, img_eye_mask


# TODO 获取到虹膜区域
def segment_iris(eye_crop, eye_mask):
    img_copy = eye_crop.copy()
    mask_coords = np.where(eye_mask != 0)
    mask_min_x = np.min(mask_coords[1])
    mask_max_x = np.max(mask_coords[1])

    eye_GrayRed = img_copy[..., 2]
    negative_mask = np.logical_not(eye_mask)
    eye_GrayRed[negative_mask] = 0

    edges = canny(eye_GrayRed, sigma=2.0, low_threshold=40, high_threshold=70)
    edges_mask = canny(eye_mask * 255, sigma=1.5, low_threshold=1, high_threshold=240)
    edges_mask = binary_dilation(edges_mask)  # 边缘膨胀
    edges_mask = np.logical_not(edges_mask)
    edges = np.logical_and(edges, edges_mask)

    diam = mask_max_x - mask_min_x
    radius_min = int(diam / 4.0)
    radius_max = int(diam / 2.0)
    hough_radii = np.arange(radius_min, radius_max, 1)
    hough_res = hough_circle(edges, hough_radii)

    # select best detection
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1, normalize=True)

    # select central point and diam/4 as fallback
    if radii is None or radii.size == 0:
        cx_glob = int(np.mean(mask_coords[1]))
        cy_glob = int(np.mean(mask_coords[0]))
        radius_glob = int(diam / 4.0)
    else:
        cx_glob = cx[0]
        cy_glob = cy[0]
        radius_glob = radii[0]

    # generate mask for iris
    iris_mask = np.zeros_like(eye_mask, dtype=np.uint8)
    cv2.circle(iris_mask, (cx_glob, cy_glob), radius_glob, 255, -1)
    iris_mask = np.logical_and(iris_mask, eye_mask)

    roi_iris_mask = np.logical_not(iris_mask)
    eye_GrayRed[roi_iris_mask] = 255

    iris_area = eye_GrayRed <= np.min(eye_GrayRed) + 50
    highlight = eye_GrayRed > 180
    area = np.logical_or(iris_area, highlight)
    area = ~area
    if np.sum(eye_GrayRed[area==1]) == 0:
        mean = 0
    else:
        mean = np.mean(eye_GrayRed[area == 1])
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(eye_GrayRed, cmap=plt.get_cmap('gray'))
    # plt.subplot(222)
    # plt.imshow(highlight, cmap=plt.get_cmap('gray'))
    # plt.subplot(223)
    # plt.imshow(area, cmap=plt.get_cmap('gray'))
    # plt.subplot(224)
    # plt.imshow(iris_area, cmap=plt.get_cmap('gray'))
    # plt.show()

    return eye_GrayRed, iris_mask, mean - 25 - np.min(eye_GrayRed)


def get_fit(pred_masks):
    pred_raw_masks = np.asarray((pred_masks.cpu() > 0).to(dtype=torch.uint8)) * 255
    pred_raw_masks = pred_raw_masks[0][0]
    contours, hierarchy = cv2.findContours(pred_raw_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pred_circle_mask = np.zeros(pred_raw_masks.shape)
    pred_circle_edge = np.zeros(pred_raw_masks.shape)
    max_area = 0
    max_contour = 0
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_contour = contour
            max_area = area
    # print(max_area)
    if max_area == 0 or max_contour.shape[0] < 5:
        return 0.0
    pred_raw_edge = np.zeros(pred_raw_masks.shape)
    cv2.drawContours(pred_raw_edge, contours, -1, 255, 2)
    ellipse_param = cv2.fitEllipse(max_contour)
    cv2.ellipse(pred_circle_edge, ellipse_param, 255, 2)
    cv2.ellipse(pred_circle_mask, ellipse_param, 255, -1)

    img_union = pred_raw_edge + pred_raw_masks
    pred_union = pred_circle_edge + pred_circle_mask
    img_union = np.where(img_union > 255, 255, 0)
    pred_union = np.where(pred_union > 255, 255, 0)
    bounryIOU = np.sum(np.logical_and(img_union, pred_union))/np.sum(np.logical_or(img_union, pred_union))
    # print('BIOU:', bounryIOU)
    return bounryIOU


def get_draw_img(pred_masks, origin_img):
    pred_raw_masks = np.asarray((pred_masks.cpu() > 0).to(dtype=torch.uint8)) * 255
    pred_raw_masks = pred_raw_masks[0][0]
    contours, hierarchy = cv2.findContours(pred_raw_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(origin_img, contours, -1, (0,0,255), 1)
    return origin_img


def linshi(mask, img):
    pred_raw_masks = np.asarray((mask.cpu() > 0).to(dtype=torch.uint8)) * 255
    pred_raw_masks = pred_raw_masks[0][0]
    mask_coords = np.where(pred_raw_masks != 0)
    mask_min_y = np.min(mask_coords[0])
    mask_max_y = np.max(mask_coords[0])
    mask_min_x = np.min(mask_coords[1])
    mask_max_x = np.max(mask_coords[1])

    roi_top = np.clip(mask_min_y, 0, img.shape[0])
    roi_bottom = np.clip(mask_max_y, 0, img.shape[0])
    roit_left = np.clip(mask_min_x, 0, img.shape[1])
    roi_right = np.clip(mask_max_x, 0, img.shape[1])

    roi_image = img[roi_top:roi_bottom, roit_left:roi_right, :]
    return roi_image