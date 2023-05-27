# this is devideDataToTrainValTest.py
#%%
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
import sys
sys.path.append('..')
from data.dataPreparationModule import dataPreparation
import torch
#%%
telcoChurn,trainTestXY_ = dataPreparation(criticalOutlierColsForARow=1)

outputSize = 1#kkk I should reshape the data in datapreparation and check if it would still work on the mlClassifiers
testToValRatio = 3
testToValCoeff = 1/(testToValRatio + 1)
trainInputs = torch.from_numpy(trainTestXY_.xTrain).float()
trainOutputs = torch.reshape(torch.from_numpy(trainTestXY_.yTrain).float(), (-1, outputSize))
testInputs = torch.from_numpy(trainTestXY_.xTest).float()
testOutputs = torch.reshape(torch.from_numpy(trainTestXY_.yTest).float(), (-1, outputSize))

valInputs = testInputs[:int(len(testInputs)*testToValCoeff)]
testInputs = testInputs[int(len(testInputs)*testToValCoeff):]
valOutputs = testOutputs[:int(len(testOutputs)*testToValCoeff)]
testOutputs = testOutputs[int(len(testOutputs)*testToValCoeff):]
#%%
#%%
