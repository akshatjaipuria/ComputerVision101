import torch


def Train(pModel, pTrainLoader, pDevice, pOptimizer, pCriterion, pCurrentEpoch):
    pModel.train()

    for batch_idx, data in enumerate(pTrainLoader):
        data["input_mnist_image"] = data["input_mnist_image"].to(pDevice)
        data["input_number"] = torch.nn.functional.one_hot(data["input_number"], num_classes=10).to(pDevice)  # converting number to one hot vector
        data["mnist_gt"] = data["mnist_gt"].to(pDevice)
        data["sum_gt"] = data["sum_gt"].to(pDevice)

        pOptimizer.zero_grad()  # making gradients 0, so that they are not accumulated over multiple batches
        mnist_dig_output, dig_sum_output = pModel(
            data["input_mnist_image"], data["input_number"]
        )

        mnist_dig_loss = pCriterion(mnist_dig_output, data["mnist_gt"])
        mnist_dig_loss.backward(retain_graph=True)  # calculating gradients

        dig_sum_loss = pCriterion(dig_sum_output, data["sum_gt"])
        dig_sum_loss.backward()  # calculating gradients

        pOptimizer.step()  # updating weights

        if batch_idx == (len(pTrainLoader) - 1):
            print(
                "Train Epoch: {}  Loss (MNIST Digit): {:.6f}\tLoss (Digit Sum): {:.6f}".format(
                    pCurrentEpoch,
                    mnist_dig_loss.item(),
                    dig_sum_loss.item()
                )
            )
            print(f"Batch ID: {batch_idx}")


def Validate(pModel, pValidLoader, pDevice, pCriterion):
    # setting model evaluate mode, takes care of batch norm, dropout etc. not required while validation
    pModel.eval()
    mnist_dig_valid_loss = 0
    dig_sum_valid_loss = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(pValidLoader):
            data["input_mnist_image"] = data["input_mnist_image"].to(pDevice)
            data["input_number"] = torch.nn.functional.one_hot(data["input_number"], num_classes=10).to(pDevice)  # converting number to one hot vector
            data["mnist_gt"] = data["mnist_gt"].to(pDevice)
            data["sum_gt"] = data["sum_gt"].to(pDevice)

            mnist_dig_output, dig_sum_output = pModel(
                data["input_mnist_image"], data["input_number"]
            )

            mnist_dig_valid_loss += pCriterion(mnist_dig_output, data["mnist_gt"]).item()
            dig_sum_valid_loss += pCriterion(dig_sum_output, data["sum_gt"]).item()

    mnist_dig_valid_loss /= len(pValidLoader)
    dig_sum_valid_loss /= len(pValidLoader)

    print(
        "Average Validation loss:  Loss (MNIST Digit): {:.6f}\tLoss (Digit Sum): {:.6f}".format(
            mnist_dig_valid_loss, 
            dig_sum_valid_loss
        )
    )


def TrainModel(pModel, pTrainLoader, pValidLoader, pCriterion, pEpochs=10, pLearningRate=0.001, pDevice="cpu"):
    optim = torch.optim.SGD(pModel.parameters(), lr=pLearningRate)
    pModel.to(pDevice)

    for epoch in range(1, pEpochs + 1):
        if pTrainLoader:
            Train(pModel, pTrainLoader, pDevice, optim, pCriterion, epoch)
        if pValidLoader:
            print("Validating.....")
            Validate(pModel, pValidLoader, pDevice, pCriterion)
