<!-- [![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.10%2B-orange)](https://www.tensorflow.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.patcog.2025.112150-blue)](https://ieeexplore.ieee.org/document/10884719#:~:text=10.1109/JSTARS.2025.3541733)
-->

[![Paper](https://img.shields.io/badge/Paper-IEEE%20JSTARS%202025-2b6cb0.svg)](#citation)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FJSTARS.2025.3541733-4a5568.svg)](#citation)
[![Tasks](https://img.shields.io/badge/Tasks-Classification-success.svg)](#-features)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-informational.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)]()

## BrownViTNet: Hybrid CNN-Vision Transformer Model for the Classification of Brownfields in Aerial Imagery


#### Abstract:
The identification of brownfield sites through satellite imagery is a crucial yet challenging classification problem in the field of remote sensing applications. In this study, we leverage aerial images sourced from Google Maps, Bing Maps, and national aerial and satellite imagery (DOP20), focusing on three distinct land use classes: active areas, construction sites, and brownfields, all matched across the same geographical coordinates. Our dataset initially includes images set against a 1000x1000 pixel blank canvas, often resulting in a significant portion of unused black space. To optimize this, images were cropped based on a threshold ensuring a minimum height and width of 400 pixels, resulting in a substantial reduction of the dataset by 42.8\% for active areas, 65.93\% for construction areas, and 75.86\% for brownfield images. Given the reduced size of our usable dataset, we employed single-image super-resolution to enhance image quality and effectively double our dataset size as a augmentation part of training data. For model architecture, the initial four layers consist of a Convolutional Neural Network (CNN), followed by intermediate layers using a Vision Transformer (ViT) with a patch size of 16. This novel hybrid architecture of BrownViTNet (Brownfield Vision Transformer Network) demonstrated impressive accuracy in the classification of brownfields from satellite imagery, presenting a significant advance in the use of machine learning for environmental monitoring and urban planning.
