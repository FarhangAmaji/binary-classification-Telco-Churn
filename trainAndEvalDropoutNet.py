#%% imports
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
from dropoutNetModule import dropoutNet
import torch
import torch.optim as optim
from dataPreparationModule import dataPreparation
#%%
telcoChurn,trainTestXY_ = dataPreparation(criticalOutlierColsForARow=1)
#%%
# Set random seed for reproducibility
torch.manual_seed(42)

# Define the hyperparameters
inputSize = trainTestXY_.xTrain.shape[1]
outputSize = 1#kkk I should reshape the data in datapreparation and check if it would still work on the mlClassifiers
dropoutRate = 0.5
learningRate = 0.0003
testToValRatio = 3
testToValCoeff = 1/(testToValRatio + 1)
numEpochs = 300#kkk early stopping
numSamples = 200
batchSize = 64

# Create the model
model = dropoutNet(inputSize, outputSize, dropoutRate)

# Define the loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

trainInputs = torch.from_numpy(trainTestXY_.xTrain).float()
trainOutputs = torch.reshape(torch.from_numpy(trainTestXY_.yTrain).float(), (-1, outputSize))
testInputs = torch.from_numpy(trainTestXY_.xTest).float()
testOutputs = torch.reshape(torch.from_numpy(trainTestXY_.yTest).float(), (-1, outputSize))
#%%
# Train the model
model.trainModel(trainInputs, trainOutputs, criterion, optimizer, numEpochs, batchSize)
#%%
# Evaluate the model with dropout
model.evaluateModel(testInputs, testOutputs, numSamples, batchSize)