# Weakly Supervised Bone Fracture Segmentation and Pseudo-Mask Generation from X-ray Images Using Deep Learning

## Introduction

Bone fracture segmentation in X-ray images is an important task in medical image analysis, helping identify, localize, and visualize fracture regions. However, developing accurate segmentation models is challenging because pixel-level annotations require significant time, effort, and expert medical knowledge.

This project proposes a **weakly supervised deep learning framework** for bone fracture localization and pseudo-mask generation using image-level labels instead of manually annotated pixel-level masks. Publicly available bone fracture X-ray datasets are used to train deep learning classifiers through transfer learning, distinguishing between fractured and non-fractured images.

To localize potential fracture regions, the framework explores **Class Activation Mapping (CAM), Grad-CAM, and other suitable localization techniques**. The resulting localization maps are processed and refined to generate binary pseudo-masks representing predicted fracture regions. Overlay images are also produced to visualize the localized regions on the original X-ray images.

The quality of the generated pseudo-masks is evaluated against ground-truth masks on an independent test set using standard segmentation metrics, including **Dice Coefficient, Intersection over Union (IoU), Precision, and Recall**. Additionally, the generated pseudo-masks are explored as potential training data for downstream segmentation models.

The primary goal of this project is to reduce dependence on expensive pixel-level annotations and support the development of scalable, data-efficient bone fracture segmentation methods.

## Key Features

- **Transfer Learning:** Train deep learning classifiers using pretrained models.
- **Weakly Supervised Localization:** Identify potential fracture regions using image-level labels.
- **Pseudo-Mask Generation:** Convert localization maps into binary fracture masks.
- **Visual Interpretation:** Generate overlay images highlighting predicted fracture regions.
- **Segmentation Evaluation:** Measure pseudo-mask quality using Dice, IoU, Precision, and Recall.
- **Downstream Segmentation:** Explore pseudo-masks for training segmentation models.

## Project Objective

To investigate whether image-level supervision combined with deep learning-based localization techniques can generate useful fracture pseudo-masks and reduce the need for manual pixel-level annotation in bone fracture segmentation.


Author:

Shibli1123
Alif-Lucifer
