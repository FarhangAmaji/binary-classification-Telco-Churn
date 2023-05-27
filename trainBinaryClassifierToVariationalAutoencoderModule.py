# this is trainBinaryClassifierToVariationalAutoencoderModule.py
#%% imports
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)

from deepLearning.variationalAutoencoderModule import variationalAutoencoder
from deepLearning.binaryClassifierToVariationalAutoencoderModule import binaryClassifierToVariationalAutoencoder, trainBinaryClassifierToVariationalAutoencoder
from deepLearning.binaryClassifierToVariationalAutoencoderModule import evaluateBinaryClassifierToVariationalAutoencoder
import torch
import torch.optim as optim
from data.devideDataToTrainValTest import trainInputs, trainOutputs, valInputs, valOutputs, testInputs, testOutputs
os.chdir(baseFolder)

vaeBestModel=torch.load(r'data\outputs\bestVaeModel')
vae=variationalAutoencoder(*vaeBestModel['inputArgs'])
vae.load_state_dict(vaeBestModel['model'])
#%%
# Set random seed for reproducibility
torch.manual_seed(42)

# Set device (CPU or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the hyperparameters
inputSize = trainInputs.shape[1]
hiddenSize = inputSize
dropoutRate = 0.95
learningRate = 0.00006
batchSize = 64
latentDim = 4
freezeOption=True

numEpochs = 300

# Extract the encoder part of the VAE
vaeEncoder = vae.encoder

if freezeOption:
    # Freeze the VAE encoder weights
    vaeEncoder.eval()

# Create the binaryClassifierToVariationalAutoencoder instance and move it to the device
bcVae = binaryClassifierToVariationalAutoencoder(vaeEncoder, hiddenSize).to(device)

# Define the binary classification loss (e.g., binary cross-entropy)
binaryClassificationLoss = torch.nn.BCELoss()

# Define the binaryClassifierToVariationalAutoencoder optimizer
binaryClassifierToVariationalAutoencoderOptimizer = optim.Adam(bcVae.parameters(), lr=learningRate, weight_decay=1e-3)

# Training loop for binary classification
numEpochs = 300
bcVae=trainBinaryClassifierToVariationalAutoencoder(bcVae, trainInputs, trainOutputs, valInputs, valOutputs, binaryClassificationLoss, binaryClassifierToVariationalAutoencoderOptimizer,
             numEpochs, batchSize, dropoutRate, device, patience=10, savePath=r'data\outputs\bestbinaryClassifierVaeModel')
#%%
evaluateBinaryClassifierToVariationalAutoencoder(bcVae, testInputs, testOutputs, batchSize, device)