# dropoutNetModule.py
import torch
import torch.nn as nn

class dropoutNet(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(dropoutNet, self).__init__()

        self.fc1 = nn.Linear(inputSize, 400)
        self.lRelu = nn.LeakyReLU(negative_slope=0.05)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, outputSize)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.lRelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.lRelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    def trainModel(self, trainInputs, trainOutputs, valInputs, valOutputs, criterion, optimizer, numEpochs, batchSize, numSamples, dropoutRate, patience, savePath):
        self.train()
        bestValAccuracy = 0.0
        counter = 0
        
        self.dropout.p = dropoutRate
        
        for epoch in range(numEpochs):
            runningLoss = 0.0
            
            for i in range(0, len(trainInputs), batchSize):
                optimizer.zero_grad()
                
                batchTrainInputs = trainInputs[i:i+batchSize]
                batchTrainOutputs = trainOutputs[i:i+batchSize]
                
                batchTrainOutputsPred = self.forward(batchTrainInputs)
                loss = criterion(batchTrainOutputsPred, batchTrainOutputs)
                
                loss.backward()
                optimizer.step()
                
                runningLoss += loss.item()
            
            epochLoss = runningLoss / (len(trainInputs) / batchSize)
            print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {epochLoss:.4f}")
            
            with torch.no_grad():
                valAccuracy = self.evaluateModel(valInputs, valOutputs, numSamples, batchSize, dropoutRate)
                
                if valAccuracy > bestValAccuracy:
                    bestValAccuracy = valAccuracy
                    counter = 0
                    torch.save(self.state_dict(), savePath)
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping! in {epoch} epoch")
                        break
        
        print("Training finished.")
        
        # Load the best model
        self.load_state_dict(torch.load(savePath))
        
        # Return the best model
        return self

    def evaluateModel(self, inputs, outputs, numSamples, batchSize, dropoutRate):
        self.eval()
        self.dropout.p = dropoutRate
        
        with torch.no_grad():
            correct = 0
            
            for i in range(0, len(inputs), batchSize):
                batchInputs = inputs[i:i+batchSize]
                batchOutputs = outputs[i:i+batchSize]
                appliedBatchSize, outputSize = batchOutputs.shape
                
                batchOutputsPred = torch.zeros((numSamples, appliedBatchSize))
                
                batchOutputsPred = torch.stack(tuple(map(lambda x: self.forward(x).squeeze(), [batchInputs] * numSamples)))
                
                meanOutput = batchOutputsPred.mean(dim=0)
                predictions = torch.reshape((meanOutput >= 0.5).float(), (-1, outputSize))
                
                correct += (predictions == batchOutputs).sum().item()
            
            accuracy = correct / len(inputs)
            print(f"Accuracy: {accuracy:.4f}")
            
            return accuracy
