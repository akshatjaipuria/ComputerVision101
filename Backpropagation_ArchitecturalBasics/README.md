# Backpropagation

This section is dedicated towards understanding how exactly does the *Backpropagation* algorithm work. This algorithm is essentially used to propagate the errors backward in the model, which in turn is used as a feedback in updating and fine-tuning the weights and decrease the loss value (calculated using the predicted output and the ground truth). Let's consider a simple network for our use case:

<p align="center">
  <img src="files/NeuralNetwork.jpg" width="700">
</p>


For our computations, we are considering *Sigmoid* activation function and *L2 loss* function.

$$A = σ = \frac{1}{(1 + e^{-x})}$$

$$E = \frac{1}{2}(T - A\_O)^2$$

Now, we have these basic equations:

| Step | Equations                     | Details                                                      |
| ---- | ----------------------------- | ------------------------------------------------------------ |
| 1    | <p align="center"><img src="files/equation_set_1.jpg" width="1000"></p> | These are the basic equations that will help us in our further computations. The activated output from each neuron is calculated by applying activation functions on the output calculated with the help of *input values* and the *weights* in each layer. Error is calculated according to the Loss function used with the help of *target values* (Ground truth) and the Model *output*. |
| 2    | <p align="center"><img src="files/equation_set_2.jpg" width="1000"></p> | We have to calculate gradients/partial derivatives of all the weights in the model w.r.t the total error. The first set of weights that we encounter while backpropagation are W<sub>5</sub>, W<sub>6</sub>, W<sub>7</sub>, W<sub>8</sub> and hence we calculate the partial derivatives for these first. Considering W<sub>5</sub> for our calculations, the same can be extrapolated to obtain values for other partial derivatives. <br /><br />Since E<sub>2</sub> does not have a dependency on W<sub>5</sub>, it is treated as a constant. Expanding the partial derivative of E<sub>1</sub> w.r.t W<sub>5</sub>, we calculate individual components which can be substituted back. |
| 3    | <p align="center"><img src="files/equation_set_3.jpg" width="1000"></p> | Combining the obtained equations, we get the partial derivative of E<sub>T</sub> w.r.t W<sub>5</sub>. Following the similar computations we also get the partial derivatives of E<sub>T</sub> w.r.t W<sub>6</sub>, W<sub>7</sub> and W<sub>8</sub>. |
| 4    | <p align="center"><img src="files/equation_set_4.jpg" width="1000"></p> | The next set of weights that we encounter while backpropagation are W<sub>1</sub>, W<sub>2</sub>, W<sub>3</sub>, W<sub>4</sub> and hence the partial derivatives for these have to be calculated next. We will calculate this w.r.t W<sub>1</sub> and the same can be extrapolated to others. For this, we need to calculate the firs component, i.e., partial derivative of E<sub>T</sub> w.r.t A_H<sub>1</sub>. Other two components can be calculated using simple partial differentiation. |
| 5    | <p align="center"><img src="files/equation_set_5.jpg" width="1000"></p> | For calculating the partial derivative of  E<sub>T</sub> w.r.t A_H<sub>1</sub>, we need to calculate partial derivative of  E<sub>1</sub> and E<sub>2 </sub> w.r.t A_H<sub>1</sub> . We are calculating for E<sub>1</sub> first and it can be calculated for E<sub>2</sub> in the similar way. Partial derivative of  E<sub>1</sub> w.r.t A_H<sub>1</sub> can be expanded and calculated by calculating the individual components as shown. |
| 6    | <p align="center"><img src="files/equation_set_6.jpg" width="1000"></p> | From the above calculation and using the similar format to calculate for others, we have the equations for partial derivate of E<sub>1</sub> and E<sub>2 </sub> w.r.t A_H<sub>1</sub> and A_H<sub>2</sub>. |
| 7    | <p align="center"><img src="files/equation_set_7.jpg" width="1000"></p> | Now, using the calculated equations and substituting them back we have the equations for partial derivative of  E<sub>T</sub> w.r.t A_H<sub>1</sub> and A_H<sub>2</sub> . |
| 8    | <p align="center"><img src="files/equation_set_8.jpg" width="1000"></p> | Using the equations mentioned in Step 4 and other computed equations, we get our final equations for the partial derivatives of E<sub>T</sub> w.r.t W<sub>1</sub>, W<sub>2</sub>, W<sub>3</sub> and W<sub>4</sub>. |

Now, all these partial derivatives of the total error w.r.t each of the weights is used in calculating the step values (updated values) of the weights after each backpropagation. For updating the weights, we also use a multiplications factor which is called the **learning rate** (η). The step values are calculated as follows:
$$W_{x}(Updated) = W_{x}(Old) - η\frac{∂E_T}{∂W_x}$$
The numeric computations can be referred from the Excel Sheet. The following initial values  were used:

| T<sub>1</sub> | T<sub>2</sub> | I<sub>1</sub> | I<sub>2</sub> | W<sub>1</sub> | W<sub>2</sub> | W<sub>3</sub> | W<sub>4</sub> | W<sub>5</sub> | W<sub>6</sub> | W<sub>7</sub> | W<sub>8</sub> |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 0.01          | 0.99          | 0.05          | 0.1           | 0.15          | 0.2           | 0.25          | 0.3           | 0.4           | 0.45          | 0.5           | 0.55          |

By altering the learning rate, the following graphs were obtained:

<p align="center">
  <img src="files/lr_graphs.jpg" width="900">
</p>


# Architectural basics

We have some basic components and concepts that are necessary to understand in order to build a good model and provide it with a good training. These are:

1. Number of layers
2. MaxPooling
3. 1x1 Convolutions
4. 3x3 Convolutions
5. Receptive Field
6. SoftMax
7. Learning Rate
8. Kernels and how do we decide the number of kernels?
9. Batch Normalization
10. Image Normalization
11. Position of MaxPooling
12. Concept of Transition Layers
13. Position of Transition Layers
14. Dropout
15. When do we introduce Dropout, or when do we know we have some overfitting
16. The distance of MaxPooling from the Prediction layer
17. The distance of Batch Normalization from Prediction layer
18. How do we know our network is not going well, comparatively, very early
19. Batch Size, and effects of batch size, etc.



The idea here is to apply all these concepts and build a model to train over MNIST dataset and achieve *more than 99.4%* Validation accuracy with less than *20K Parameters* and *within 20 epochs*.



While performing these exercise, a few **observations** were made (we are referring to validation accuracy everywhere below):

1. The convergence of the accuracy was very fast till 98-99%
2. Even the a lighter model (with less than 3-4K parameters) could easily be trained to achieve 98-99% accuracy.
3. Given X number of parameters and a good crafting of model, a model which is deeper with approximately X no. of parameters performed much better than if we would just get the parameter count high by increasing the no. of channels.
4. There were few images in the validation dataset that the model was actually struggling to predict correctly:

<p align="center">
  <img src="files/difficult_inputs.png">
</p>


After observing the incorrectly predicted images, certain augmentations like **Center crop** and **Random rotation** were used to mimic the rough data while training. The model architecture and the training can be referred from the `ArchitecturalBasics.ipynb` notebook.

1. Model Parameter Count: 12,696
2. Total training epochs: 20
3. Highest Validation Accuracy: 99.5%

Loss and accuracy curve:

<p align="center">
  <img src="files/loss_accuracy_curve.jpg", width="700">
</p>

