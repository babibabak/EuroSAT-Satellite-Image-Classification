# EuroSAT-Satellite-Image-Classification
A satellite image classification system for the EuroSAT dataset, built with TensorFlow to classify 64x64 RGB images into 10 land-use categories (e.g., AnnualCrop, Forest, River). The project loads the dataset, splits it into 70% training, 15% validation, and 15% test sets, and inspects dataset properties.
# EuroSAT Satellite Image Classification
## Overview
This project implements a satellite image classification system using the EuroSAT dataset, which contains 27,000 64x64 RGB satellite images across 10 land-use categories (e.g., AnnualCrop, Forest, Highway, SeaLake). The project loads the dataset using TensorFlow Datasets, splits it into 70% training, 15% validation, and 15% test sets, and inspects dataset properties such as the number of classes and samples. It is designed to support convolutional neural network (CNN) models for image classification, with preprocessing and visualization capabilities. The project uses Python with libraries like TensorFlow Datasets, TensorFlow, NumPy, and Matplotlib.

## Features
- Data Loading: Loads the EuroSAT dataset (RGB images) using TensorFlow Datasets.
- Data Splitting: Divides the dataset into 70% training (18,900 samples), 15% validation (4,050 samples), and 15% test (4,050 samples) sets.
- Dataset Inspection: Extracts and displays the number of classes (10), class names, and total number of samples (27,000).
-  Visualization: Supports visualization of sample images using TensorFlow Datasets' visualization tools (extendable with Matplotlib).
- Extensibility: Provides a foundation for building and training CNN models for land-use classification.

## Dataset
The EuroSAT dataset, available via TensorFlow Datasets, includes:
- Total Samples: 27,000 RGB satellite images.
- Image Dimensions: 64x64 pixels with 3 color channels (RGB).
- Classes: 10 land-use categories: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.
- Splits:
  - Training: 18,900 images (70%).
  - Validation: 4,050 images (15%).
  - Test: 4,050 images (15%).

## Requirements
- Python 3.11
- Libraries: `tensorflow`, `tensorflow-datasets`, `numpy`, `matplotlib`

## Installation
1.Clone the repository:
```bash
git clone https://github.com/yourusername/eurosat-classification.git
```
2.Install dependencies:
```bash
pip install tensorflow tensorflow-datasets numpy matplotlib
```
3.Run the Jupyter Notebook:
```bash
jupyter notebook final-project.ipynb
```

## Usage
-Load the EuroSAT dataset using TensorFlow Datasets.
- Split the dataset into training (70%), validation (15%), and test (15%) sets.
- Inspect dataset properties, including the number of classes (10), class names, and total samples (27,000).
- Visualize sample images using TensorFlow Datasets' visualization tools.
- (Optional) Extend the project by implementing CNN models for classification and evaluating performance on the test set.

## Example
The project loads the EuroSAT dataset and splits it as follows:
- Training Set: 18,900 images.
- Validation Set: 4,050 images.
- Test Set: 4,050 images.
- Classes: 10 (e.g., AnnualCrop, Forest, River).Sample images can be visualized to inspect the dataset, and the code is structured to support further model development.

## Methodology
- Data Loading: Uses TensorFlow Datasets to load the EuroSAT dataset (RGB split).
- Data Splitting: Manually splits the single train split into training (70%), validation (15%), and test (15%) sets using take and skip operations.
- Inspection: Extracts metadata (number of classes, class names, total samples) using dataset info.
- Visualization: Leverages TensorFlow Datasets' visualization tools to display sample images (extendable with Matplotlib).
- Future Steps: The project is designed to support CNN model development, training with the Adam optimizer, and evaluation using metrics like accuracy and F1 score.

## Future Improvements
- Implement convolutional neural networks (CNNs) with layers like Conv2D, MaxPooling, and Dense for improved classification accuracy.
- Apply data augmentation (e.g., rotations, flips) to enhance model robustness.
- Evaluate models using metrics like accuracy, F1 score, and confusion matrices.
- Add hyperparameter tuning with tools like Keras Tuner to optimize model performance.
- Enhance visualization with custom Matplotlib plots for training history and class distributions.
