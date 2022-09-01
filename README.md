## KinectPy
@author: Tibor Camargo

Point cloud processing and analysis of data extracted from multiple Azure Kinect sensors. Includes point cloud registration, filtering, human segmentation and pose estimation.

## Disclaimer
Currently a personal project, it will be available as a PyPi project around 12/2022 with detailed documentation, installation steps and with a Docker container for reproducing experiments.


![Workflow_Cropped](https://user-images.githubusercontent.com/25236592/184530784-173c3fab-c608-4ea5-8021-25115c98f32e.png)

## Dependencies

There are external dependencies that have not been included yet, such as PyMoCapViewer and OfflineProcessor. 

If you are using this software for point cloud processing after MKV extraction:
* Mask R-CNN is used for human segmentation, please [download Mask R-CNN weights and configuration here](https://drive.google.com/drive/folders/1lBcEwQ45tMbKLQOJxIykWy2D5roPO4cH). Do not forget to pass the path to the `.pb` and `.pbtxt` accordingly in the `extract.py`.

## Installation

Currently not available
