# Advanced training concepts

This time, we will be training a ResNet18 model over CIFAR-10 dataset and also use GradCam to understand what the model is trying to see in order to predict a particular class.

But what is **GradCam** in simple terms?

ChatGPT explains it as :

GradCam is a technique for visualizing the regions in an image that contribute the most to a particular prediction made by a Convolutional Neural Network (CNN). It's used to better understand how CNNs make predictions, especially in cases where they make incorrect predictions.

GradCam works by backpropagating the gradients of the prediction with respect to the activations of a chosen layer, and then weighting the activations of each channel by these gradients. The resulting values are then up-sampled to the original image size and overlaid on the image, highlighting the regions that have the most influence on the prediction.

In simpler terms, GradCam uses the gradients of the prediction to highlight the areas of the image that are most relevant to the prediction made by the CNN, making it easier to see why the network is making the predictions it is.

## Training and results

Our dataset looks like below after applying CutOut and RandomCrop augmentations:

![dataset](files/dataset.png)

We trained for 20 Epochs and achieved a maximum validation accuracy of 83.45% and maximum training accuracy of 75.31%.

The accuracy and  loss curves are as follows:

![acc](files/acc.png)

![loss](files/loss.png)

Some misclassified images:

![misclassified](files/misclassified.png)

### GradCam results

![grad1](files/grad1.png)

![grad2](files/grad2.png)

![grad3](files/grad3.png)

![grad4](files/grad4.png)

![grad5](files/grad5.png)

![grad6](files/grad6.png)

![grad7](files/grad7.png)

![grad8](files/grad8.png)

![grad9](files/grad9.png)

![grad10](files/grad10.png)