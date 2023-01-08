import torch


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def Train(pModel, pTrainLoader, pDevice, pOptimizer, pCriterion):
    pModel.train()
    mnist_dig_total_loss = 0
    dig_sum_total_loss = 0
    mnist_dig_correct_pred = 0
    dig_sum_correct_pred = 0

    for batch_idx, data in enumerate(pTrainLoader):
        data["input_mnist_image"] = data["input_mnist_image"].to(pDevice)
        data["input_number"] = torch.nn.functional.one_hot(data["input_number"], num_classes=10).to(pDevice)  # converting number to one hot vector
        data["mnist_gt"] = data["mnist_gt"].to(pDevice)
        data["sum_gt"] = data["sum_gt"].to(pDevice)

        pOptimizer.zero_grad()  # making gradients 0, so that they are not accumulated over multiple batches
        mnist_dig_output, dig_sum_output = pModel(
            data["input_mnist_image"], data["input_number"]
        )

        # calculating loss
        mnist_dig_loss = pCriterion(mnist_dig_output, data["mnist_gt"])
        dig_sum_loss = pCriterion(dig_sum_output, data["sum_gt"])

        # accumulating loss over batches
        mnist_dig_total_loss += mnist_dig_loss.item()
        dig_sum_total_loss += dig_sum_loss.item()

        # accumulating no. of correct predictions over batches
        mnist_dig_correct_pred += GetCorrectPredCount(mnist_dig_output, data["mnist_gt"]) 
        dig_sum_correct_pred += GetCorrectPredCount(dig_sum_output, data["sum_gt"])

        # calculating gradients
        mnist_dig_loss.backward(retain_graph=True)  
        dig_sum_loss.backward()

        pOptimizer.step()  # updating weights

    print(
        "Training: (MNIST Digit: Loss = {:.6f}, Acc = {:.2f}%)\t(Digit Sum: Loss = {:.6f}, Acc = {:.2f}%)".format(
            mnist_dig_total_loss / len(pTrainLoader),
            100. * (mnist_dig_correct_pred / len(pTrainLoader.dataset)),
            dig_sum_total_loss / len(pTrainLoader),
            100. * (dig_sum_correct_pred / len(pTrainLoader.dataset))
        )
    )


def Validate(pModel, pValidLoader, pDevice, pCriterion):
    # setting model evaluate mode, takes care of batch norm, dropout etc. not required while validation
    pModel.eval()
    mnist_dig_valid_loss = 0
    dig_sum_valid_loss = 0
    mnist_dig_correct_pred = 0
    dig_sum_correct_pred = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(pValidLoader):
            data["input_mnist_image"] = data["input_mnist_image"].to(pDevice)
            data["input_number"] = torch.nn.functional.one_hot(data["input_number"], num_classes=10).to(pDevice)  # converting number to one hot vector
            data["mnist_gt"] = data["mnist_gt"].to(pDevice)
            data["sum_gt"] = data["sum_gt"].to(pDevice)

            mnist_dig_output, dig_sum_output = pModel(
                data["input_mnist_image"], data["input_number"]
            )

            # accumulating loss over batches
            mnist_dig_valid_loss += pCriterion(mnist_dig_output, data["mnist_gt"]).item()
            dig_sum_valid_loss += pCriterion(dig_sum_output, data["sum_gt"]).item()

            # accumulating no. of correct predictions over batches
            mnist_dig_correct_pred += GetCorrectPredCount(mnist_dig_output, data["mnist_gt"])
            dig_sum_correct_pred += GetCorrectPredCount(dig_sum_output, data["sum_gt"])

    print(
        "Validation:  (MNIST Digit: Loss = {:.6f}, Acc = {:.2f}%)\t(Digit Sum: Loss = {:.6f}, Acc = {:.2f}%)".format(
            mnist_dig_valid_loss / len(pValidLoader), 
            100. * (mnist_dig_correct_pred / len(pValidLoader.dataset)),
            dig_sum_valid_loss / len(pValidLoader),
            100. * (dig_sum_correct_pred / len(pValidLoader.dataset))
        )
    )


def TrainModel(pModel, pTrainLoader, pValidLoader, pCriterion, pEpochs=10, pLearningRate=0.001, pDevice="cpu"):
    pModel.to(pDevice)
    optim = torch.optim.Adam(pModel.parameters(), lr=pLearningRate)

    for epoch in range(1, pEpochs + 1):
        print(f"-------------- Epoch {epoch} --------------")
        if pTrainLoader:
            Train(pModel, pTrainLoader, pDevice, optim, pCriterion)
        if pValidLoader:
            Validate(pModel, pValidLoader, pDevice, pCriterion)
