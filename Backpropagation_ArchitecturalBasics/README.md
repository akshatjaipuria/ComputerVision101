# Backpropagation

This section is dedicated towards understanding how exactly does the *Backpropagation* algorithm work. This algorithm is essentially used to propagate the errors backward in the model, which in turn is used as a feedback in updating and fine-tuning the weights and decrease the loss value (calculated using the predicted output and the ground truth). Let's consider a simple network for our use case:

<p align="center">
  <img src="files/NeuralNetwork.jpg" width="700">
</p>
For our computations, we are considering *Sigmoid* activation function and L2 loss function.
$$
A = σ = \frac{1}{(1 + e^{-x})}
$$ {Sigmoid Activation function}

$$
E = \frac{1}{2}(T - A\_O)^2
$$ {L2 Loss}

Now, we have these basic equations:

| Step | Equations                     | Details                                                      |
| ---- | ----------------------------- | ------------------------------------------------------------ |
| 1    | ![](files\equation_set_1.jpg) | These are the basic equations that will help us in our further computations. The activated output from each neuron is calculated by applying activation functions on the output calculated with the help of *input values* and the *weights* in each layer. Error is calculated according to the Loss function used with the help of *target values* (Ground truth) and the Model *output*. |
| 2    | ![](files\equation_set_2.jpg) | We have to calculate gradients/partial derivatives of all the weights in the model w.r.t the total error. The first set of weights that we encounter while backpropagation are W<sub>5</sub>, W<sub>6</sub>, W<sub>7</sub>, W<sub>8</sub> and hence we calculate the partial derivatives for these first. Considering W<sub>5</sub> for our calculations, the same can be extrapolated to obtain values for other partial derivatives. <br /><br />Since E<sub>2</sub> does not have a dependency on W<sub>5</sub>, it is treated as a constant. Expanding the partial derivative of E<sub>1</sub> w.r.t W<sub>5</sub>, we calculate individual components which can be substituted back. |
| 3    | ![](files\equation_set_3.jpg) | Combining the obtained equations, we get the partial derivative of E<sub>T</sub> w.r.t W<sub>5</sub>. Following the similar computations we also get the partial derivatives of E<sub>T</sub> w.r.t W<sub>6</sub>, W<sub>7</sub> and W<sub>8</sub>. |
| 4    | ![](files\equation_set_4.jpg) | The next set of weights that we encounter while backpropagation are W<sub>1</sub>, W<sub>2</sub>, W<sub>3</sub>, W<sub>4</sub> and hence the partial derivatives for these have to be calculated next. We will calculate this w.r.t W<sub>1</sub> and the same can be extrapolated to others. For this, we need to calculate the firs component, i.e., partial derivative of E<sub>T</sub> w.r.t A_H<sub>1</sub>. Other two components can be calculated using simple partial differentiation. |
| 5    | ![](files\equation_set_5.jpg) | For calculating the partial derivative of  E<sub>T</sub> w.r.t A_H<sub>1</sub>, we need to calculate partial derivative of  E<sub>1</sub> and E<sub>2 </sub> w.r.t A_H<sub>1</sub> . We are calculating for E<sub>1</sub> first and it can be calculated for E<sub>2</sub> in the similar way. Partial derivative of  E<sub>1</sub> w.r.t A_H<sub>1</sub> can be expanded and calculated by calculating the individual components as shown. |
| 6    | ![](files\equation_set_6.jpg) | From the above calculation and using the similar format to calculate for others, we have the equations for partial derivate of E<sub>1</sub> and E<sub>2 </sub> w.r.t A_H<sub>1</sub> and A_H<sub>2</sub>. |
| 7    | ![](files\equation_set_7.jpg) | Now, using the calculated equations and substituting them back we have the equations for partial derivative of  E<sub>T</sub> w.r.t A_H<sub>1</sub> and A_H<sub>2</sub> . |
| 8    | ![](files\equation_set_8.jpg) | Using the equations mentioned in Step 4 and other computed equations, we get our final equations for the partial derivatives of E<sub>T</sub> w.r.t W<sub>1</sub>, W<sub>2</sub>, W<sub>3</sub> and W<sub>4</sub>. |

Now, all these partial derivatives of the total error w.r.t each of the weights is used in calculating the step values (updated values) of the weights after each backpropagation. For updating the weights, we also use a multiplications factor which is called the **learning rate** (η). The step values are calculated as follows:
$$
W_{x}(Updated) = W_{x}(Old) - η\frac{∂E_T}{∂W_x}
$$
The numeric computations can be referred from the Excel Sheet. Using certain initial values and altering the learning rate, the following graphs were obtained:

| T1   | T2   | I1   | I2   | W1   | W2   | W3   | W4   | W5   | W6   | W7   | W8   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0.01 | 0.99 | 0.05 | 0.1  | 0.15 | 0.2  | 0.25 | 0.3  | 0.4  | 0.45 | 0.5  | 0.55 |

<p align="center">
  <img src="files/lr_graphs.jpg" width="900">
</p>



# Architectural basics
