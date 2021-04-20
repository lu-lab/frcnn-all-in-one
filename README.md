# custom faster r-cnn in your browser
---
This set of notebooks serves as documentation and tutorial for preparing data and training a Faster R-CNN inception v2 object detection model entirely in web-based notebooks. This means you don't have to install everything to your own computer to prep your data and train a deep learning model, it's all automatically installed in the notebooks contained in this repository!

There are two notebooks to work through in order to train a Faster R-CNN with your own custom classes. First, you will need to annotate images and save the annotation data by working through the Faster R-CNN annotation notebook. Second, you'll use those annotations to train the model using the Google colab notebook in the 'colab' folder in this repo.

### Annotate your data
Click this binder badge to run the annotation notebook from your browser and follow the directions in Faster-RCNN-annotation.ipynb: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lu-lab/frcnn-all-in-one/HEAD)
**Make sure to download label_map.pbtxt and bounding_boxes.csv before you close the binder!**

### Train your model
Once you have your annotations, train the model in Google Colab so that it can be trained in a reasonable length of time. 
Click this colab badge to view the Faster R-CNN training notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lu-lab/frcnn-all-in-one/blob/main/colab/Faster_R_CNN_training.ipynb)
However, instead of running this notebook directly, copy it to your own Google Drive by choosing File→Save a copy in Drive. Once you've saved this in your own Google Drive, the first cell you run will automatically clone this repo and keep just the other files within the colab folder that you will need to follow the training notebook.

This notebook is associated with our [preprint](https://www.biorxiv.org/content/10.1101/2021.02.08.430359v1). 
