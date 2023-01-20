# Coding Drilldown

In this part, our target is to achieve:

1. Consistent 99.4% validation accuracy
2. Within 15 Epochs
3. In less than 10k Parameters

Moreover, this has to progressed across iterations of improvements mentioning the target, results and proper analysis in each iteration. (Minimum of 3 iterations are expected)

## Iteration 1

Target:

1. Basic Set-up
2. Set Transforms
3. Set Data loader
4. Set basic working code
5. Set up training & test Loop

Results:

1. Parameters: 547k
2. Best Training Accuracy: 99.28
3. Best Test Accuracy: 99.03

Analysis:

1. Model is too heavy for the given problem and is also overfitting
2. Validation accuracy seem to decrease in later epochs

## Iteration 2

Target:

1. Get a good and lighter model skeleton, which should be stable and not require constant changes
2. Add Batch normalization to increase efficiency
3. Using a GAP layer
4. Hereafter, the training has to be concluded within 15 epochs

Results:

1. Parameters: 7924
2. Best Training Accuracy: 99.31
3. Best Test Accuracy: 98.95

Analysis:

1. Model seems to be pretty good in terms of skeleton and the parameters
2. Model is overfitting on the training data
3. Also seems like there's a lot of difficult data that model struggles to learn/predict

## Iteration 3

Target:

1. Add dropout for regularization
2. Add transforms to improve the training as well as for regularization

Results:

1. Parameters: 7924
2. Best Training Accuracy: 98.72
3. Best Test Accuracy: 99.30

Analysis:

1. Model is not overfitting at all
2. We are very close to our target of 99.4 % Validation accuracy
3. The model's convergence is slow, improving it will help achieve our target

## Iteration 4

Target:

1. Use LR scheduler
2. Experiment with a higher learning rate along with LR scheduler to fasten up convergence

Results:

1. Parameters: 7924
2. Best Training Accuracy: 99.12
3. Best Test Accuracy: 99.42 (14th Epoch)

Analysis:

1. Model seems to perform well consistently during the final epochs.
2. Finding a good LR schedule is hard, We have tried to make it learn faster and effectively by reducing LR after 10th epoch.
3. We did achieve our target of 99.4 % accuracy, but an accuracy of 99.5 % is still achievable with an even better training. 
