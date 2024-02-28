# Gesture2Text: Real-Time Sign Language Translation Using Deep Learning

## Overview

This project presents Gesture2Text, a real-time sign language translation system that utilizes deep learning to bridge communication gaps between the deaf and hearing communities. It harnesses the power of convolutional neural networks to recognize hand gestures from webcam input and translate them into corresponding text, fostering seamless and inclusive communication.

## Key Features

- Real-time gesture recognition and translation: Enables effortless sign language interpretation in real-time, facilitating natural and - fluid conversations.
- High accuracy: Trained on the comprehensive Sign Language MNIST dataset, the model achieves exceptional accuracy in gesture recognition, ensuring reliable translation.
- User-friendly interface: The system is designed with simplicity and ease of use in mind, making it accessible to individuals with varying levels of technical expertise.
- Robust and adaptable: Built using Keras and OpenCV, Gesture2Text offers a solid foundation for further enhancements and customization to meet diverse user needs.

## Future Vision

- Word and Sentence Translation: Building upon the robust CNN architecture, the project's potential lies in expanding its vocabulary to translate words and even sentences, empowering more comprehensive communication.
- Enhanced Gesture Recognition: Incorporating techniques like hand tracking and contextual understanding can improve accuracy and robustness in capturing complex gestures.
- Multilingual Support: Expanding languages will broaden the project's reach and inclusivity.


## Model Architecture

Convolutional Neural Network (CNN) with:
Two convolutional layers with max pooling
Dropout for regularization
Batch normalization
A fully connected layer with 128 neurons
A final softmax layer for classification
## Training

Trained on Google Colab using Keras with ImageDataGenerator for data augmentation.
Achieved an accuracy of 99.4% on training set in average and 85.6% on test set.

## Sample images

![1 (1)](https://github.com/sharathkumaar/Gesture2Text/assets/112824465/1d9ad1ea-adca-47ba-b119-1ef7b06e9e29)
![3](https://github.com/sharathkumaar/Gesture2Text/assets/112824465/69e80551-906b-484d-87d8-9c42c1a0bf56)
![2](https://github.com/sharathkumaar/Gesture2Text/assets/112824465/9d719e64-3e3f-4e6b-a210-51719f3d9ea6)
![4](https://github.com/sharathkumaar/Gesture2Text/assets/112824465/25f327c2-e0f2-41ec-9bb6-badfff5f386d)



## Getting Started


### Install Required Libraries

```bash
pip install -r requirements.txt
```

## Download the Project

Clone this repository or download the zip file.

```bash
git clone [https://github.com/sharathkumaar/Gesture2Text.git]

```

## Download the Dataset:

Visit the Kaggle website and download the Sign Language MNIST dataset then extract the contents to project root directory.

Sign Language MNIST: https://www.kaggle.com/datasets/datamunge/sign-language-mnist


## Train the Model:

- Open the train.ipynb notebook in Jupyter Notebook or Google Colab.
- Follow the instructions within the notebook to execute the training process.
- This will create the trained model file (CNNmodel.h5).

## Run the Script

```bash
python main.py
```
Execute the main code That will launch the real-time prediction interface using your webcam.


**Contributing:**

We welcome contributions! Please feel free to submit pull requests or open issues for any enhancements or bug fixes.

**License:**

This project is licensed under the MIT License. See the LICENSE file for details.
