#%% imports
import os
import pandas as pd
base_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_folder)

from dataPreparationModule import dataPreparation
from modelEvaluatorModule import modelEvaluator,trainTestXY
from modelConfigs import allModelConfigss
from utils import q,ti

from sklearn.preprocessing import MinMaxScaler
#%% 
telcoChurn, xTrain, xTest, yTrain, yTest = dataPreparation(doUpSampling=False, criticalOutlierColsForARow=1)
defaultCrossValidationNum = 5#kkk change it back to 5
scaler = MinMaxScaler()
xTrain, xTest, yTrain, yTest=scaler.fit_transform(xTrain), scaler.fit_transform(xTest), scaler.fit_transform(yTrain), scaler.fit_transform(yTest)
trainTestXY_ = trainTestXY(xTrain, xTest, yTrain, yTest)
#%%
t0=ti()
csvFileName='noUpsamplingChurnTotResultsDf5cv'
if os.path.exists(f'{csvFileName}.csv'):
    totResultsDf = pd.read_csv(f'{csvFileName}.csv')
else:    
    totResultsDf = pd.DataFrame()
if __name__ == '__main__':
    for m1 in allModelConfigss:
        print(m1.name,'started',ti()-t0,'s')
        totResultsDf = pd.concat([totResultsDf, m1.fitModelAndGetResults(trainTestXY_,totResultsDf,parallel=True)])
        totResultsDf.to_csv(f'{csvFileName}.csv', index=False, header=True)
        print(m1.name,'finished',ti()-t0,'s')
    print('lasted',ti()-t0,'s')
#%% 

#%% 

#%% 

#%% 

#%% 

