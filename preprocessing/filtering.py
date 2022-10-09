import imghdr
import os
import cv2
import copy
import logging
import numpy as np
import open3d as o3d
import tensorflow as tf
from PIL import Image


def filter_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 200, 
    std_ratio: float = 3.0,
    voxel_size: float = 0.02
    ) -> o3d.geometry.PointCloud:
    """ 
    Applies a statistical outlier removal for the Point Cloud `pcd`.
    This method is specially useful to be applied before using any 
    procedure that involves Oriented Bounding Boxes
    """
    voxel_down_pcd = copy.deepcopy(pcd).voxel_down_sample(voxel_size)
    cloud, _  = voxel_down_pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return cloud


class Filtering:
    def __init__(self, frozen_graph_fp, pbtxt_fp):
        self.net = self._load_segmentation_model(frozen_graph_fp, pbtxt_fp)

    def _load_segmentation_model(self, frozen_graph_fp, pbtxt_fp):
        lib_dir = os.getcwd()
        if not frozen_graph_fp:
            frozen_graph_fp = os.path.join(lib_dir, 'data', 'frozen_inference_graph.pb')
        if not pbtxt_fp:
            pbtxt_fp = os.path.join(lib_dir, 'data', 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt') 
        
        assert os.path.isfile(frozen_graph_fp), "Invalid path to mask-rcnn frozen graph"
        assert os.path.isfile(pbtxt_fp), "Invalid path to mask-rcnn pbtxt"

        net = cv2.dnn.readNetFromTensorflow(frozen_graph_fp, pbtxt_fp)
        return net


    def apply_segmentation(self, img):
        """ 
        It will apply a segmentation model where the class
        representing a human is labeled as 15, but I will convert it to 255 
    
        Args:
            img (np.array): filepath to image    
        Returns:
            segmentation map
        """
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        self.net.setInput(blob)
        boxes, masks = self.net.forward(['detection_out_final', 'detection_masks'])

        person_id = 0
        black_image = np.zeros_like(img)
        height, width, _ = img.shape
        detected_boxes = boxes.shape[2]

        for i in range(detected_boxes):
            # Selecting the i-th box
            box = boxes[0, 0, i] 
            class_id = int(box[1])
            score = box[2]

            if class_id == person_id and score >= 0.5:

                # Getting box coordinates
                x1 = int(box[3] * width)
                y1 = int(box[4] * height)
                x2 = int(box[5] * width)
                y2 = int(box[6] * height)

                roi = black_image[y1:y2, x1:x2]
                roi_height, roi_width, _ = roi.shape

                # Selecting the mask associated to ith box
                mask = masks[i, class_id]
                mask = cv2.resize(mask, (roi_width, roi_height))
                mask[mask < 0.1] = 0
                mask[mask >= 0.1] = 255

                # Get mask coordinates
                contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cv2.fillPoly(roi, [cnt], (255, 255, 255))
     
        mask_idx = np.argwhere(black_image != 255)
        img[mask_idx[:, 0], mask_idx[:, 1], mask_idx[:, 2]] = black_image[mask_idx[:, 0], mask_idx[:, 1], mask_idx[:, 2]]
        return img


def kalman_filter(joint_vals: np.ndarray, ri=10, qi=10, fi=1/30, hi=1) -> np.ndarray:
    ''' 
    Kalman filtering used for correcting self-occlusions 

    It is applied on a matrix of `joint_vals` with N observations and 3 dimensions, 
    '''
    
    N = len(joint_vals)
    
    Pi = np.identity(3)
    Fi = fi*np.identity(3)
    Ri = ri*np.identity(3)
    Qi = qi*np.identity(3)
    Hi = hi*np.identity(3)
    xhi = joint_vals[0]
    
    x_preds = [xhi]
    for i in range(1, N):

        # Prediction step:
        xhdi = Fi@xhi
        Pdi = Fi@Pi@Fi.T + Qi

        # Update step:
        Ki = Pdi@Hi.T@np.linalg.inv(Hi@Pdi@Hi.T + Ri)
        xhi = xhdi + Ki@(joint_vals[i] - Hi@xhdi)
        Pi = (np.identity(3) - Ki@Hi)@Pdi

        x_preds.append(xhi)

    x_preds = np.array(x_preds)
    return x_preds
