# Real-Time Violence Detection

This repository contains a deep learning-based system for real-time violence detection using surveillance video streams. The system classifies live video input as *violent* or *non-violent* and raises alerts whenever violent activity is detected.

## Overview

The project is designed to assist in monitoring environments such as schools, workplaces, and public spaces by leveraging computer vision and deep learning techniques. It consists of two main components:

1. **Model Generator**

   * Preprocesses and transforms the dataset.
   * Extracts features and labels from video frames.
   * Resizes images and trains the model.
   * Saves the trained model for inference.

2. **Model Tester**

   * Loads the trained model.
   * Captures live camera feed.
   * Detects and classifies real-time activities as violent or non-violent.
   * Raises system alerts when violence is detected.

## Dataset

The project uses the **RWF-2000 dataset**, which consists of:

* 1000 violent videos
* 1000 non-violent videos

Dataset link: [RWF-2000 Video Database](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)

The dataset must be downloaded manually and placed in the project directory before training the model.

## Technologies Used

* Python
* TensorFlow/Keras
* MobileNetV2 (Transfer Learning)
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Flask

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/renuka2113/Real-Time-Violence-Detector.git
cd Real-Time-Violence-Detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python model_generator.py
```

### 4. Run the Model on Live Feed

```bash
python model_tester.py
```
