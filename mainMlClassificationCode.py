#%% imports
import os
import pandas as pd
import numpy as np
base_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_folder)

from dataPreparationModule import dataPreparation
from modelEvaluatorModule import modelEvaluator
from modelConfigs import allModelConfigs
from envVarsPreprocess import envVars
from utils import q,ti, inputTimeout
#%% 
telcoChurn, trainTestXY_ = dataPreparation(criticalOutlierColsForARow=1)
#%%
t0=ti()
inputTimeout(f'have u checked the envVars.\nenvVars={envVars}',30)
csvFileName=envVars['csvFileName']
if os.path.exists(f'{csvFileName}.csv'):
    totResultsDf = pd.read_csv(f'{csvFileName}.csv')
    #kkk I can check for the file lock(file being used by other programs) by resaving it
else:    
    totResultsDf = pd.DataFrame()
if __name__ == '__main__':
    for m1 in allModelConfigs:
        print(m1.name,'started',ti()-t0,'s')
        totResultsDf = pd.concat([totResultsDf, m1.fitModelAndGetResults(trainTestXY_,totResultsDf,parallel=envVars['parallel'])]).reset_index(drop=True)
        totResultsDf.to_csv(f'{csvFileName}.csv', index=False, header=True)
    print('lasted',ti()-t0,'s')
#%% 

#%% 

#%% 

#%% 

#%% 

