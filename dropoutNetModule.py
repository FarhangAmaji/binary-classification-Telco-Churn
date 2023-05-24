import torch
import torch.nn as nn

class dropoutNet(nn.Module):
    def __init__(self, inputSize, outputSize, dropoutRate):
        super(dropoutNet, self).__init__()
        self.dropoutRate = dropoutRate
        
        self.fc1 = nn.Linear(inputSize, 400)
        self.lRelu = nn.LeakyReLU(negative_slope=0.05)
        self.dropout = nn.Dropout(p=self.dropoutRate)
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

    def trainModel(self, inputs, outputs, criterion, optimizer, numEpochs, batchSize):
        self.train()
        
        for epoch in range(numEpochs):
            runningLoss = 0.0
            
            for i in range(0, len(inputs), batchSize):
                optimizer.zero_grad()
                
                batchInputs = inputs[i:i+batchSize]
                batchOutputs = outputs[i:i+batchSize]
                
                batchOutputsPred = self.forward(batchInputs)
                loss = criterion(batchOutputsPred, batchOutputs)
                
                loss.backward()
                optimizer.step()
                
                runningLoss += loss.item()
            
            epochLoss = runningLoss / (len(inputs) / batchSize)
            print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {epochLoss:.4f}")
        
        print("Training finished.")

    def evaluateModel(self, inputs, outputs, numSamples, batchSize):
        self.eval()
        
        with torch.no_grad():
            correct = 0
            
            for i in range(0, len(inputs), batchSize):
                batchInputs = inputs[i:i+batchSize]
                batchOutputs = outputs[i:i+batchSize]
                appliedBatchSize, outputSize=batchOutputs.shape
                
                batchOutputsPred = torch.zeros((numSamples, appliedBatchSize))
                
                batchOutputsPred = torch.stack(tuple(map(lambda x: self.forward(x).squeeze(), [batchInputs] * numSamples))                )
                
                meanOutput = batchOutputsPred.mean(dim=0)
                predictions = torch.reshape((meanOutput >= 0.5).float(), (-1, outputSize))
                
                correct += (predictions == batchOutputs).sum().item()
            
            accuracy = correct / len(inputs)
            print(f"Accuracy: {accuracy:.4f}")
