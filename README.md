# Attention Detection in Online Education Using Spectral Image Mapping and Deep Learning

This repository contains the official implementation of the hybrid approach proposed for detecting cognitive focus and confusion in online learning environments using Electroencephalography (EEG) signals.

Instead of processing noisy, raw time-series data, this methodology converts EEG signals into frequency-based spectral images via Fast Fourier Transform (FFT) and classifies them using a 2D Convolutional Neural Network (CNN).

## Overview

In online education, tracking students' real-time cognitive state and focus level can be challenging. This project addresses the limitation by providing a computationally efficient, high-performance hybrid architecture suitable for real-time integration. 

* **Data Augmentation:** Overlapping Windowing is applied to handle class imbalance.
* **Feature Extraction:** 1D time-series signals are transformed via FFT into 64x14 2D spectral images, capturing critical energy shifts in Alpha (8-12 Hz) and Beta (13-30 Hz) bands.
* **Classification:** A tailored 3-block 2D-CNN architecture yields competitive performance with minimal computational cost compared to heavy LSTM or complex signal processing methods.

## Performance

The model was validated on the publicly available Mental Attention State Detection (Inan Cigdem) Dataset, recorded via a 14-channel Emotiv EPOC+ device at 128 Hz.

* **Test Accuracy:** 96.46%
* **Recall (Focused/Attentive Class):** 1.00
* **Precision (Focused/Attentive Class):** 0.94

This demonstrates that the network effectively identifies focused states without missing critical attention windows, outperforming traditional machine learning models (SVM, KNN) and resource-heavy hybrid models (CNN+LSTM).

## System Architecture

The pipeline consists of four main phases:
1. **Preprocessing and Filtering:** 0.5-40 Hz Butterworth band-pass filter to remove artifacts.
2. **Data Balancing:** Overlapping Windowing (2-step shift for the "Attentive" class; 128-step shift for the "Normal" class).
3. **Spectral Imaging:** Applying FFT, taking the first 64 positive frequency points, log transformation, and Standard Scaler normalization to build 64x14 matrices.
4. **CNN Model:** Three consecutive Conv2D blocks (32, 64, and 128 filters) with Batch Normalization, ReLU activation, MaxPooling2D, and Dropout (0.3 - 0.4), leading to a Dense layer and a Sigmoid output.

## Repository Structure

* `preprocessing/`: Scripts for Butterworth filtering and Overlapping Windowing.
* `features/`: FFT transformation and spectral image generation code.
* `models/`: 2D-CNN architecture definition and training pipeline.
* `notebooks/`: Exploratory data analysis and training demonstrations.

## Requirements

* Python 3.x
* NumPy
* SciPy
* TensorFlow / Keras
* Matplotlib
* Scikit-learn

## Future Work

* Expanding the classification task to include complex emotional states such as "Bored", "Stressed", and "Excited".
* Integrating Explainable AI (XAI) techniques like Grad-CAM to visualize which brain regions (Frontal, Occipital, etc.) drive the classification decisions.
* Developing a live prototype combined with a web camera interface to provide instructors with a real-time "Classroom Attention Heatmap".

## Citation

If you find this work useful in your research, please cite the original study:

Gumusboga, M. F., Akyol, S., & Ozyurt, F. Çevrimiçi Eğitimde Bireylerin Dikkat Tespiti İçin Spektral Özellik Görüntüleme ve Derin Öğrenme Tabanlı Hibrit Bir Yaklaşım. Firat University, Faculty of Engineering, Software Engineering.
