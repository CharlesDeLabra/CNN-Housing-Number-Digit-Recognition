# CNN-Housing-Number-Digit-Recognition
<p align="center">
    <img src="https://github.com/CharlesDeLabra/Housing-Number-recognition-with-ANN/blob/main/images/number.png?raw=true" alt="Logo" width=72 height=72>
  <h3 align="center">Convolutional Neural Networks: Street View Housing Number Digit Recognition</h3>
  <p align="center">
    This project solved a classification problem of Digit recognition to classify the housing number of a house. It was trained convolutional Neural Networks for solving the problem.
    <br>
  </p>
</p>

## Table of contents

- [Context](#context)
- [Problem Statement](#problem-statement)
- [Code](#code)
- [Status](#status)
- [Analysis](#analysis)
- [Conclusions](#conclusions)
- [Recomendations](#recomendations)

## Context

One of the most interesting tasks in deep learning is to recognize objects in natural scenes. The ability to process visual information using machine learning algorithms can be very useful as demonstrated in various applications.

The SVHN dataset contains over 600,000 labeled digits cropped from street level photos. It is one of the most popular image recognition datasets. It has been used in neural networks created by Google to improve map quality by automatically transcribing the address numbers from a patch of pixels. The transcribed number with a known street address helps pinpoint the location of the building it represents.

## Problem Statement

There is a huge dataset of housing numbers and it is needed to train a model that allow the system to classify the digits in order to recognize which digits is on each images. Since the digits are stored in images, it is needed to use NN so it will be check if CNN solves the problem.

## Code

The program was written on Jupyter Notebooks in Python Language. You can access to the code [here](https://github.com/CharlesDeLabra/CNN-Housing-Number-Digit-Recognition/blob/main/CNN_Project_Learner_Notebook_SVHN.ipynb)

## Status

The code is finished and have been evaluated, the goal was completed since the model was good enough to predict 80% of the digits but it fails in some digits compared to other ones and the results were better than the ones from ANN.

## Analysis

The first steps was importing the data and after that it was necessary to see the images, so it was printed the first 10 images in order to visualize out dataset:

<br>
<p align="center">
    <img src="https://github.com/CharlesDeLabra/Housing-Number-recognition-with-ANN/blob/main/images/data.png?raw=true" alt="Data" width=911 height=100> 
</p>
<br>

After that it was needed to preprocess the data, so the next steps were followed:

- Print the first image in the train image and figure out the shape of the images
- Reshape the train and the test dataset to flatten them. Figure out the required shape
- Normalise the train and the test dataset by dividing by 255
- Print the new shapes of the train and the test set
- One-hot encode the target variable

After completing these steps it were created two models, the first one stucture was the next one: 

- First Convolutional layer with 16 filters and kernel size of 3x3. Use the 'same' padding and provide an apt input shape
- LeakyRelu layer with the slope equal to 0.1
- Second Convolutional layer with 32 filters and kernel size of 3x3 with 'same' padding
- LeakyRelu with the slope equal to 0.1
- A max-pooling layer with a pool size of 2x2
- Flatten the output from the previous layer
- dense layer with 32 nodes
- LeakyRelu layer with slope equal to 0.1
- final output layer with nodes equal to the number of classes and softmax activation 

The Model was compiled with the categorical_crossentropy loss, adam optimizers (learning_rate = 0.001), and accuracy metric. After creating and training the previous model the results were the next ones:

<br>
<p align="center">
    <img src="https://github.com/CharlesDeLabra/CNN-Housing-Number-Digit-Recognition/blob/main/images/data2.png?raw=true" alt="Data2" width=630 height=626> 
</p>
<br>

The results were not acceptable since the gap between both results were high and that tell the model is overfitting, so the second model was created:

- First Convolutional layer with 16 filters and kernel size of 3x3. Use the 'same' padding and provide an apt input shape
- LeakyRelu layer with the slope equal to 0.1
- Second Convolutional layer with 32 filters and kernel size of 3x3 with 'same' padding
- LeakyRelu with the slope equal to 0.1
- Max-pooling layer with a pool size of 2x2
- BatchNormalization layer
- Third Convolutional layer with 32 filters and kernel size of 3x3 with 'same' padding
- LeakyRelu layer with slope equal to 0.1
- Fourth Convolutional layer 64 filters and kernel size of 3x3 with 'same' padding
- LeakyRelu layer with slope equal to 0.1
- Max-pooling layer with a pool size of 2x2
- BatchNormalization layer
- Flatten the output from the previous layer
- Dense layer with 32 nodes
- LeakyRelu layer with slope equal to 0.1
- Dropout layer with rate equal to 0.5
- Final output layer with nodes equal to the number of classes and softmax activation

The model was compiled with the categorical_crossentropy loss, adam optimizers (learning_rate = 0.001), and accuracy metric. Do not fit the model here, just return the compiled model. After creating and training the model, the results were the next ones: 

<br>
<p align="center">
    <img src="https://github.com/CharlesDeLabra/CNN-Housing-Number-Digit-Recognition/blob/main/images/data3.png?raw=true" alt="Data3" width=630 height=626> 
</p>
<br>

These results were better than the previous since it were almost of 90% and this model was choosen for making our classification. The results of the classification were the next one:

<br>
<p align="center">
    <img src="https://github.com/CharlesDeLabra/CNN-Housing-Number-Digit-Recognition/blob/main/images/data4.png?raw=true" alt="Data4" width=637 height=941> 
</p>
<br>

Using this final model lead to the next parts which are conclusions and recommendations.

## Conclusions

- All of the classes have a higher f1 score more than 80%
- The model predicts better 0 and 7 with 90%
- The model worse predicts 3 with 82%
- The model confuse high 98 time 6 and 8
- Same happend with 1 and 4 with 83 times
- Same happens with 5 and 3 with 114 times
- The rest are decent results but it is needed to check te ones who are near 40 times

## Recomendations

- The best model for predicting houses was created from CNN
- The second model was the best one
- It could be improved the solution for certain digits.
