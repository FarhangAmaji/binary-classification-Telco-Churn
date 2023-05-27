# this is trainVariationalAutoencoder.py
#%% imports
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
import torch
import torch.nn as nn
import torch.optim as optim
from deepLearning.variationalAutoencoderModule import variationalAutoencoder, trainVae, evaluateVae
from data.devideDataToTrainValTest import trainInputs, trainOutputs, valInputs, valOutputs, testInputs, testOutputs
os.chdir(baseFolder)
#%% 
# Set random seed for reproducibility
torch.manual_seed(42)

# Set device (CPU or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the hyperparameters
inputSize = trainInputs.shape[1]
outputSize = 1
dropoutRate = 0.95
learningRate = 0.000006
batchSize = 64
latentDim= 4

# Create VAE instance and move it to the device
vae = variationalAutoencoder(inputSize, latentDim).to(device)

# Define reconstruction loss
reconstructionLoss = torch.nn.MSELoss()

# Define optimizer
vaeOptimizer = optim.Adam(vae.parameters(), lr=learningRate, weight_decay=1e-3)

# Training loop
numEpochs = 300
vae=trainVae(vae, trainInputs, trainOutputs, valInputs, valOutputs, reconstructionLoss, vaeOptimizer, numEpochs,
             batchSize, dropoutRate, device, patience=10, savePath=r'data\outputs\bestVaeModel')
#%%
from deepLearning.variationalAutoencoderModule import variationalAutoencoder, trainVae, evaluateVae
vaeBestModel=torch.load(r'data\outputs\bestVaeModel')
vae=variationalAutoencoder(*vaeBestModel['inputArgs'])
vae.load_state_dict(vaeBestModel['model'])