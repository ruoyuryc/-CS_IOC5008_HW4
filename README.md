# CS_IOC5008_HW4
**Code for "VRDL homework 4: Tiny PASCAL VOC dataset instance segmentation" in National Chiao Tung University**
## Hardware
 - Tesla K40m  
## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#Dataset-Preparation)
5. [Train and inference](#Train-and-inference)
6. [Make Submission](#make-submission)


## Installation
Clone this repository:
```git clone https://github.com/ruoyuryc/CS_IOC5008_HW4```</br>
Install dependencies:
```pip3 install -r requirements.txt```</br>
Run setup from the repository root directory:
```python3 setup.py install```</br>
## Dataset Preparation
download image on [google drive](https://drive.google.com/drive/u/3/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK)
### Prepare Images
After downloading, the data directory is structured as:
```
train data
  +- train_image
  +- pascal_train.json
test data
  +- test_image
  +- test.json
```

1349 training image and 100 testing image collected form voc dataset with 20 classes</br>
### Dataset location
put the data under CS_IOC5008_HW4/datasets
## Train and Inference
### Train a new model starting from ImageNet weights and inference
```
$ python3 dataset/tinyVOC.py 
```
### step by step training and inference
* [VOCtiny.ipynb](samples/demo.ipynb)  Is the easiest way to start. It shows how to train Mask R_CNN model pre-trained on imagenet to segment objects in this homework dataset. It includes code to run object detection and instance segmentation on arbitrary images.

修改的Mask R-CNN
https://github.com/aihill/Mask_RCNN_Keras
(因為直接使用Matterport的Mask R-CNN會有問題)

## Make Submission


## Other refenence
modified Mask R-CNN:https://github.com/aihill/Mask_RCNN_Keras
[http://www.immersivelimit.com/tutorials/using-mask-r-cnn-on-custom-coco-like-dataset]
[https://inclass.kaggle.com/hmendonca/mask-rcnn-with-submission/code]
[https://github.com/NovatecConsulting/SemanticSegmentation-Examples/blob/master/RSNA%20Pneumonia%20Detection%20(TF%20with%20Mask%20R-CNN)/TransferLearning_MaskRCNN.ipynb]
[https://inclass.kaggle.com/drt2290078/mask-rcnn-sample-starter-code]


## Citation
Use this bibtex to cite this repository:
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
