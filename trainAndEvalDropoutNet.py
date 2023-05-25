# trainAndEvalDropoutNet.py
#%% imports
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
from dropoutNetModule import dropoutNet, trainModel, evaluateModel
import torch
import torch.optim as optim
from dataPreparationModule import dataPreparation
#%%
telcoChurn,trainTestXY_ = dataPreparation(criticalOutlierColsForARow=1)
#%%
#best result 0.7328897338403042 valAcc=0.7457: dropoutRate = 1, learningRate = 0.000006, in 2nd epoch
#kkk add certainty to classification
# Set random seed for reproducibility
torch.manual_seed(42)

# Define the hyperparameters
inputSize = trainTestXY_.xTrain.shape[1]
outputSize = 1#kkk I should reshape the data in datapreparation and check if it would still work on the mlClassifiers
dropoutRate = 0.95
learningRate = 0.000006
testToValRatio = 3
testToValCoeff = 1/(testToValRatio + 1)
numEpochs = 300
numSamples = 200
batchSize = 64
#kkk add cuda for model and data
# Create the model
model = dropoutNet(inputSize, outputSize)

# Define the loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3)

trainInputs = torch.from_numpy(trainTestXY_.xTrain).float()
trainOutputs = torch.reshape(torch.from_numpy(trainTestXY_.yTrain).float(), (-1, outputSize))
testInputs = torch.from_numpy(trainTestXY_.xTest).float()
testOutputs = torch.reshape(torch.from_numpy(trainTestXY_.yTest).float(), (-1, outputSize))

valInputs = testInputs[:int(len(testInputs)*testToValCoeff)]
testInputs = testInputs[int(len(testInputs)*testToValCoeff):]
valOutputs = testOutputs[:int(len(testOutputs)*testToValCoeff)]
testOutputs = testOutputs[int(len(testOutputs)*testToValCoeff):]
#%%
# Train the model
model = trainModel(model, trainInputs, trainOutputs, valInputs, valOutputs, criterion, optimizer, numEpochs, batchSize, numSamples, dropoutRate=dropoutRate, patience=10, savePath=r'data\outputs\bestDropoutEnsembleModel')
#%%
# Evaluate the model with dropout
evaluateModel(model, testInputs, testOutputs, numSamples, batchSize, dropoutRate=dropoutRate)
