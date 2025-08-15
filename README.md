# Face Emotion Recognition using CNN (FER-2013)

## Overview

This project develops a convolutional neural network (CNN) to recognize human facial emotions from images. The model is trained and evaluated on the FER-2013 dataset, predicting one of seven emotions per image. This technology has applications in human–computer interaction (HCI), user analytics and assistive systems.

## Dataset

  * **Source:** FER-2013 (Kaggle: “Face Expression Recognition Dataset”)
  * **Size:** 35k+ grayscale face images
  * **Resolution:** 48x48 pixels
  * **Classes (7):** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

 **Download the Dataset:**
   * Download the `FER-2013` dataset from [Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data).

## Key Contributions

  * Implemented a custom CNN tailored for 48x48 grayscale inputs.
  * Utilized data augmentation and regularization to improve model generalization.
  * Achieved **61.87%** test accuracy on 7-class facial expression recognition.

## Model Architecture

The architecture is a compact CNN designed for low-resolution images:

  * **Input:** 48x48x1
  * **Layers:**
      * Multiple convolutional blocks with 3x3 kernels and ReLU activation.
      * Max-pooling layers to downsample feature maps.
      * Dense layers for classification.
      * Dropout layers for regularization.
  * **Output:** Softmax layer with 7 classes.

A representative sketch of the architecture is:
`Conv(32, 3x3) -> ReLU -> MaxPool -> Conv(64, 3x3) -> ReLU -> MaxPool -> Flatten -> Dense(128) -> Dropout -> Dense(7) -> Softmax`

## Training Setup

  * **Optimizer:** Adam
  * **Loss:** Categorical cross-entropy
  * **Metrics:** Accuracy, Precision, Recall, F1-score
  * **Data Augmentation:** Random rotations, horizontal flips, and zoom/shifts.
  * **Regularization:** Dropout in dense layers.

## Results

  * **Test Accuracy:** 61.87%
  * **Observations:**
      * "Happy" and "Surprise" classes typically achieve high performance.
      * "Fear" and "Sad" are more challenging due to visual subtlety and class imbalance.
      * Data augmentation and dropout significantly improve generalization.



## Limitations and Future Improvements

  * **Limitations:**
      * The FER-2013 dataset has label noise and class imbalance.
      * 48x48 grayscale images limit the detail available for classification.
  * **Potential Improvements:**
      * **Transfer Learning:** Fine-tune pre-trained models like VGG or MobileNet.
      * **Advanced Architectures:** Integrate residual connections or attention mechanisms.
      * **Better Data:** Use a higher-resolution or more balanced dataset.
      * **Loss Functions:** Use class-weighted or Focal Loss to address imbalance.
