## KinectPy
@author: Tibor Camargo

Point cloud processing and analysis of data extracted from multiple Azure Kinect sensors. Includes point cloud registration, filtering, human segmentation and pose estimation.

## Disclaimer
Currently a personal project, it will be available as a PyPi project around 12/2022 with detailed documentation, installation steps and with a Docker container for reproducing experiments.


![Workflow_Cropped](https://user-images.githubusercontent.com/25236592/184530784-173c3fab-c608-4ea5-8021-25115c98f32e.png)

## Dependencies

If you are using this software for point cloud processing, after MKV extraction:
* Mask R-CNN is used for human segmentation, so please download `frozen_inference_graph.pb` and `mask_rcnn_inception_v2_coco_2018_01_28.pbtxt`
from this link: https://drive.google.com/drive/folders/1lBcEwQ45tMbKLQOJxIykWy2D5roPO4cH, and set its path accordingly in the `extract.py`.

## Installation
