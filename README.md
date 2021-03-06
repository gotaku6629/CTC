
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)  

# No Blank Connectionist Temporal Classification
Pytorch implementation for [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf), but my model does not consider unvoicedness and deals with multi labels.

Contact: Takuya Goto (goto@cv.nitech.ac.jp)

## Overview

### Connectionist Temporal Classification 
<p align="center">
<img src='imgs/ctc.png' width="1000px"/>

Connectionist Temporal Classification(CTC) is Loss Function used in speech recognition and time series data. When estimating the label probability for each time from each unit of the LSTM in chronological order, the probability of including a blank character representing unvoiced sound is estimated.


### CTC for Human Action Prediction
<p align="center">
<img src='imgs/action.png' width="500px"/>
<p align="center">
<img src='imgs/ctc_action.png' width="1000px"/>   

Unlike speech recognition, there is no "nothing" state in a person's actions. So, I designed no blank CTC. In addition, since the action recognition of a person is multi-labeled, the design is suitable for it.


## Usage

### Prerequisites
- Python 3.6
- [Pytorch and torchvision](https://pytorch.org/)

### Datasets
- [Charades dataset](http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar)
- [Charades' Training and Validation Annotations](http://ai2-website.s3.amazonaws.com/data/Charades.zip)

### Pretrained Model
- Download the [pretrained (with Kinetics Dataset) I3D](https://arxiv.org/abs/1705.07750) model [here](https://www.dropbox.com/s/r6ja11h06y2w83c/rgb_i3d_pretrained.pt?dl=0). Note that I removed the last classifier layer, and append a new classifier layer for Charades.

### Run the Code
1. Modify CTC_exe.py
    * Create the cache directory
    * Specify the location of the data, training/validation split, and pretrained model.
2. Command as follows
~~~~
python3 CTC_exe.py
~~~~
   

### Acknowledgement
A large portion of the code comes from the [ctc](https://github.com/vadimkantorov/ctc) and [ImageNet](https://github.com/pytorch/examples/tree/master/imagenet) repo.
