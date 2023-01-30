# Batch Normalization & Regularization

Here, we will be training a model using three different normalization techniques, Group Normalization, Layer Normalization and Batch Normalization. The losses and accuracies have to be compared for all three trainings.

Note: While training the model with Batch Norm. we will also be adding the L1 regularization term to the loss.

## Group vs Layer vs Batch Normalizations

<p align="center">
  <img src="files/normalization.jpg" width="700">
</p>

## Loss

<p align="center">
  <img src="files/loss.jpg" width="700">
</p>

## Accuracies

<p align="center">
  <img src="files/acc.jpg" width="700">
</p>

## Misclassifications

| Group Norm.                            | Layer Norm.                            | Batch Norm.                            |
| -------------------------------------- | -------------------------------------- | -------------------------------------- |
| <img src="files/gn_misclassified.jpg"> | <img src="files/ln_misclassified.jpg"> | <img src="files/bn_misclassified.jpg"> |

