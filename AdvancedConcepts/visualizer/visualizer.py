import matplotlib.pyplot as plt
import numpy as np
import math


def Unnorm(pTensor, pMean, pStd):

    for t, m, s in zip(pTensor, pMean, pStd):
        np.multiply (t, s, out = t)
        np.add (t, m, out = t)
        # The normalize code -> t.sub_(m).div_(s)
    return pTensor


def DisplayData(pDataLoader, pClassLabelsMap=None, pMeanStd=None):
    batch_data, batch_label = next(iter(pDataLoader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        img = batch_data[i].numpy()
        if pMeanStd:
            img = Unnorm(img, pMeanStd[0], pMeanStd[1])
        plt.imshow(np.transpose(img, (1, 2, 0)))
        if pClassLabelsMap:
            plt.title(pClassLabelsMap[batch_label[i]])
        else:
            plt.title(batch_label[i])
        plt.xticks([])
        plt.yticks([])


def DisplayIncorrectPredictions(pIncPredDict, pClassLabelsMap=None, pMeanStd=None):
    fig = plt.figure(figsize=(2, 5))

    num_imgs = min(10, len(pIncPredDict["images"]))
    for i in range(num_imgs):
        plt.subplot(5, math.ceil(num_imgs / 5), i + 1)
        img = np.array(pIncPredDict["images"][i])
        if pMeanStd:
            img = Unnorm(img, pMeanStd[0], pMeanStd[1])
        plt.imshow(np.transpose(img, (1, 2, 0)))
        if pClassLabelsMap:
            plt.title(
                f"GT: {pClassLabelsMap[pIncPredDict['ground_truths'][i]]}\nP: {pClassLabelsMap[pIncPredDict['predicted_vals'][i]]}",
                fontsize=8,
            )
        else:
            plt.title(
                f"GT: {pIncPredDict['ground_truths'][i]}, P: {pIncPredDict['predicted_vals'][i]}",
                fontsize=8,
            )
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


def PlotTestVsTrain(pTrainInfoLabelAndList: tuple, pTestInfoLabelAndList: tuple):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(pTrainInfoLabelAndList[1])
    axs[0].set_title(pTrainInfoLabelAndList[0])
    axs[1].plot(pTestInfoLabelAndList[1])
    axs[1].set_title(pTestInfoLabelAndList[0])
