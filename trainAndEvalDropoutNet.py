# trainAndEvalDropoutNet.py
#%% imports
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
from deepLearning.dropoutNetModule import dropoutNet, trainModel, evaluateModel
import torch
import torch.optim as optim
from data.devideDataToTrainValTest import trainInputs, trainOutputs, valInputs, valOutputs, testInputs, testOutputs
os.chdir(baseFolder)
#%%
#best result 0.7328897338403042 valAcc=0.7457: dropoutRate = 1, learningRate = 0.000006, in 2nd epoch
#kkk add certainty to classification
# Set random seed for reproducibility
torch.manual_seed(42)

# Set device (CPU or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the hyperparameters
inputSize = trainInputs.shape[1]
outputSize = 1
dropoutRate = 0.95
learningRate = 0.000006
numEpochs = 300
numSamples = 200
batchSize = 64

# Create the model
model = dropoutNet(inputSize, outputSize).to(device)

# Define the loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3)
#%%
# Train the model
model = trainModel(model, trainInputs, trainOutputs, valInputs, valOutputs, criterion, optimizer, numEpochs, batchSize, numSamples, dropoutRate=dropoutRate, device=device, patience=10, savePath=r'data\outputs\bestDropoutEnsembleModel')
#%%
# Evaluate the model with dropout
evaluateModel(model, testInputs, testOutputs, numSamples, batchSize, device=device, dropoutRate=dropoutRate)
