import matplotlib.pyplot as plt
import math


def DisplayData(pDataLoader):
    batch_data, batch_label = next(iter(pDataLoader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap="gray")
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


def DisplayIncorrectPredictions(pIncPredDict):
    fig = plt.figure(figsize=(2, 5))

    num_imgs = min(10, len(pIncPredDict["images"]))
    for i in range(num_imgs):
        plt.subplot(5, math.ceil(num_imgs / 5), i + 1)
        plt.imshow(pIncPredDict["images"][i][0], cmap="gray_r")
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
