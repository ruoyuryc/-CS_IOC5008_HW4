# CS_IOC5008_HW4-Tiny PASCAL VOC dataset instance segmentation
code for VRDL homework 4 in National Chiao Tung University
## Hardware
 - Tesla K40m  
## Reproducing Submission

## Installation
Clone this repository:
```git clone https://github.com/ruoyuryc/CS_IOC5008_HW4```
Install dependencies:
```pip3 install -r requirements.txt```
Run setup from the repository root directory:
```python3 setup.py install```
## Dataset Preparation
download image on [google drive](https://drive.google.com/drive/u/3/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK)
### Prepare Images
```
train data
  +- train_image
  +- pascal_train.json
test data
  +- test_image
  +- test.json
```



## Train and inference

### Train a new model starting from ImageNet weights and inference
python3 dataset/tinyVOC.py 


### step by step training and inference
* [VOCtiny.ipynb](samples/demo.ipynb)  Is the easiest way to start. It shows an example of using a model pre-trained on MS COCO to segment objects in your own images. It includes code to run object detection and instance segmentation on arbitrary images.

修改的Mask R-CNN
https://github.com/aihill/Mask_RCNN_Keras
(因為直接使用Matterport的Mask R-CNN會有問題)


參考資料:
http://www.immersivelimit.com/tutorials/using-mask-r-cnn-on-custom-coco-like-dataset


augmentation:
http://www.immersivelimit.com/tutorials/using-mask-r-cnn-on-custom-coco-like-dataset
https://inclass.kaggle.com/hmendonca/mask-rcnn-with-submission/code
https://github.com/NovatecConsulting/SemanticSegmentation-Examples/blob/master/RSNA%20Pneumonia%20Detection%20(TF%20with%20Mask%20R-CNN)/TransferLearning_MaskRCNN.ipynb
https://inclass.kaggle.com/drt2290078/mask-rcnn-sample-starter-code
