# Cats vs Dogs classifier using Tensorflow Convnet

The Dataset contains images of dogs and cats. Each image in the dataset has a dimension of 64x64. Using Tensorflow we train the Convolutional neural network on these images to classiy the images into dogs or cats.

 The training set consists of 4000 images of dogs and 4000 images of cats.

 The test set consists of 1000 images of dogs and 1000 images of cats.

The model consists of 2 convolutional layers with maxpool layer inserted in between.We flatten the feature maps generated by the second convolutional layer and pass them through the fully connected layer with 128 units or neurons to get the final probabilities in the output layer.

The model was trained for 25 epochs , with a batch size of 32 images using `adam` optimizer and `binary_crossentropy` as the loss function. The training time took around 5 minutes and the model acheived an accuracy of 81%

Try out the model by giving your own images by adding it to the ```single_prediction``` folder and changing the name of the file in ```Making a single prediction``` section in the code and see what the model predicts.

The Model can be trained to classify any two classes of images since its a Binary Classifier, and experiment with the hyperparameters to get high accuracies.
