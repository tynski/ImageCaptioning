#!/bin/sh
#get train data
if [ ! -d "./coco/train2014/" ]
then
  mkdir ./coco/train2014/
  gsutil -m rsync gs://images.cocodataset.org/train2014 ./coco/train2014
fi
#get val data
if [ ! -d "./coco/val2014/" ]
then
  mkdir ./coco/val2014
  gsutil -m rsync gs://images.cocodataset.org/val2014 ./coco/val2014
fi
#get test data
if [ ! -d "./coco/test2014/" ]
then
  mkdir ./coco/test2014
  gsutil -m rsync gs://images.cocodataset.org/test2014 ./coco/test2014
fi
#get train/val annotations
if [ ! -d "./coco/ann2014/" ]
then
  mkdir ./coco/ann2014
  cd ./coco/ann2014
  curl -LO  http://images.cocodataset.org/annotations/annotations_trainval2014.zip
  unzip annotations_trainval2014.zip
  curl -LO  http://images.cocodataset.org/annotations/image_info_test2014.zip
  unzip image_info_test2014.zip
  rm annotations_trainval2014.zip image_info_test2014.zip
fi
#preapre folders for extracted features
if [ ! -d "./coco/train2014_features/" ]
then
  mkdir ./coco/train2014_features
fi

if [ ! -d "./coco/val2014_features/" ]
then
  mkdir ./coco/val2014_features
fi

mv ./coco/ann2014/annotations/image_info_test2014.json ./coco/ann2014/annotations/captions_test2014.json
