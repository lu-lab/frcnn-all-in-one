# custom faster r-cnn in your browser
---
This set of notebooks serves as documentation and tutorial for preparing data and training a Faster R-CNN inception v2 object detection model entirely in web-based notebooks. This means you don't have to install everything to your own computer to prep your data and train a deep learning model, it's all automatically installed in the notebooks contained in this repository! It will also work regardless of your computer's operating system (Mac/Windows/Linux).

There are two notebooks to work through in order to train a Faster R-CNN with your own custom classes. First, you will need to annotate images and save the annotation data by working through the Faster R-CNN annotation notebook. Second, you'll use those annotations to train the model using the Google colab notebook in the 'colab' folder in this repo.

### Annotate your data
Click this binder badge to run the annotation notebook from your browser and follow the directions in Faster-RCNN-annotation.ipynb: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lu-lab/frcnn-all-in-one/HEAD)
Note that the binder service may take several minutes to start. Once it starts, open the Faster-RCNN-annotation.ipynb file.  Be aware that the binder service will stop if you're inactive for 10 minutes, so make sure once you start annotating to run through the entire notebook so that you can create all the files you need and download them! It’s ok to annotate in multiple sessions, but you’ll need both the bounding_boxes.csv file and the label_map.pbtxt file for each session.
**Make sure to download label_map.pbtxt and bounding_boxes.csv before you close the binder!**

### Train your model
Once you have your annotations, train the model in Google Colab (this way you will have access to a GPU which will significantly shorten training time).
Click this colab badge to view the Faster R-CNN training notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lu-lab/frcnn-all-in-one/blob/main/colab/Faster_R_CNN_training.ipynb)
However, instead of running this notebook directly, copy it to your own Google Drive by choosing File→Save a copy in Drive. Once you've saved this in your own Google Drive, the first cell you run will automatically clone this repo and keep just the files within the colab folder that you will need to follow the training notebook.

This notebook is associated with our [preprint](https://www.biorxiv.org/content/10.1101/2021.02.08.430359v1). 
