# **KiTS19** Challenge using ***U-net***
### Image segmentation task with **KiTS19** challenge data using **U-net**

You can find details about what this challenge is, and what Data are used, in this [Official Website](https://kits19.grand-challenge.org/).


Since Data are very big (about 27GB) you have to download it from [Official data source](https://github.com/neheller/kits19), where

you can find step-by-step procedure to download dataset.
- - -
## Objective

- The goal of the challenge is **kidney** and **kidney tumor** semantic segmentation.
- While our project aims to just **kidney** segmentation. But you can easily modify code to segment **kidney tumor** as well.

## Data Preview
<img src="./images/00265.png" width="300" height="300">
(red represents kidney and blue represents tumor)

 In the segmentation, label 0 represents background, 1 represents kidney, and 2 represents tumor. But as I mentioned, sice we are
 
 interested in kidney segmentation, we replaced label 2 with 1.
 
 ## Model Structure
 
 For U-Net structure, we consulted [Official Paper](https://arxiv.org/abs/1505.04597). But we made slight changes.
 
 1. Batch Normalization after each Convolutional Layer
 2. Use padding in Convolutional Layer such that cropping isn't necessary while concatenation
 3. Drop out at the lowest part of network to avoid over-fitting
 4. Use different loss function, namely Generalized Dice Loss which was introduced in the paper [Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/abs/1707.03237)
 5. Not using data augmentation
 
 <img src="./images/UNet_structure.png" width="900" height="600">
