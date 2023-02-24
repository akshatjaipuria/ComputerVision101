import torch
from tqdm import tqdm

def EmptyFunc(*pArgs, **pKwargs):
    return 0

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def FillWrongPredictions (pData, pTarget, pPred, pOutDict):
    pred = pPred.argmax(dim=1)
    correct_pred_indices = pred.eq(pTarget) # list of Trues and Falses, Trues being at indices of correct predictions
    data = pData.tolist()
    target = pTarget.tolist()
    pred = pred.tolist()
    pOutDict['images'] += [data[x] for x in range(len(correct_pred_indices)) if not correct_pred_indices[x]]
    pOutDict['ground_truths'] += [target[x] for x in range(len(correct_pred_indices)) if not correct_pred_indices[x]]
    pOutDict['predicted_vals'] += [pred[x] for x in range(len(correct_pred_indices)) if not correct_pred_indices[x]]

def L1RegularizationTermFunc(pModel, pLambda):
    return pLambda * sum(p.abs().sum() for p in pModel.parameters())

def Train(pModel, pTrainLoader, pDevice, pOoptimizer, pCriterion, pRegularizationTermFunc, pTrainAccList, pTrainLossList, pScheduler=None, pLRHistory=None):
    pModel.train()
    pbar = tqdm(pTrainLoader)
    
    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(pDevice), target.to(pDevice)
        pOoptimizer.zero_grad()
        
        # Predict
        pred = pModel(data)
        
        # Calculate loss
        loss = pCriterion(pred, target)
        loss = loss + pRegularizationTermFunc(pModel, 0.001)
        train_loss+=loss.item()
        
        # Backpropagation
        loss.backward()
        pOoptimizer.step()
        
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)
        
        if pScheduler:
            pbar.set_description(desc= f'Train: LR={pScheduler.get_last_lr()[0]:0.6f} Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        else:
            pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        if pScheduler:
            if pLRHistory is not None:
                pLRHistory.append(pScheduler.get_last_lr()[0])
            pScheduler.step()
    
    pTrainAccList.append(100*correct/processed)
    pTrainLossList.append(train_loss/len(pTrainLoader))

def Test(pModel, pTestLoader, pDevice, pCriterion, pTestAccList, pTestLossList, pFillIncorrectSamplesFunc, pTestIncorrectPredList):
    pModel.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in pTestLoader:
            data, target = data.to(pDevice), target.to(pDevice)

            output = pModel(data)
            test_loss += pCriterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)

            pFillIncorrectSamplesFunc (data, target, output, pTestIncorrectPredList)

    test_loss /= len(pTestLoader.dataset)
    pTestAccList.append(100. * correct / len(pTestLoader.dataset))
    pTestLossList.append(test_loss)

    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(pTestLoader.dataset),
        100. * correct / len(pTestLoader.dataset)))


def TrainModel(pModel, pTrainLoader, pTestLoader, pCriterion, pOptimizer, pL1=False, pEpochs=10, pDevice="cpu", pScheduler=None):

    train_acc = []
    train_losses = []
    test_acc = []
    test_losses = []
    lr_history = [] if pScheduler else None
    test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

    reg_term_func = L1RegularizationTermFunc if pL1 else EmptyFunc

    for epoch in range(1, pEpochs + 1):
        print(f"-------------- Epoch {epoch} --------------")
        if pTrainLoader:
            Train(pModel, pTrainLoader, pDevice, pOptimizer, pCriterion(), reg_term_func, train_acc, train_losses, pScheduler, lr_history)
        if pTestLoader:
            incorrect_pred_func = FillWrongPredictions if epoch==pEpochs else EmptyFunc
            Test(pModel, pTestLoader, pDevice, pCriterion(reduction='sum'), test_acc, test_losses,  incorrect_pred_func, test_incorrect_pred)

    return {"train_acc": train_acc, "train_loss": train_losses, "test_acc": test_acc, "test_loss": test_losses, "incorrect_pred": test_incorrect_pred, "lr_history": lr_history}
