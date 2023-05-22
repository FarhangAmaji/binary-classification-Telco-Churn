#%% imports
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
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
# inputTimeout(f'have u checked the envVars.\nenvVars={envVars}',30)#kkk temp off
csvFileName=envVars['csvFileName']
if os.path.exists(f'{csvFileName}.csv'):
    totResultsDf = pd.read_csv(f'{csvFileName}.csv')
    #kkk I can check for the file lock(file being used by other programs) by resaving it
else:
    totResultsDf = pd.DataFrame()
inputArgsPool=[]
if __name__ == '__main__':
    for m1 in allModelConfigs:
        inputArgsPool.extend(m1.getInputArgs(trainTestXY_,totResultsDf))
    if envVars['parallel']:
        with ProcessPoolExecutor(max_workers=max(multiprocessing.cpu_count() - 4, 1)) as executor:
            resultRows = []
            for args in inputArgsPool:
                future = executor.submit(args[0].processFold, *args[1:])
                resultRows.append(future.result())
            totResultsDf = pd.concat([totResultsDf, *resultRows]).reset_index(drop=True)
    else:
        for ir in inputArgsPool:
            resultRows = ir[0].processFold(*ir[1:])
            totResultsDf = pd.concat([totResultsDf, resultRows]).reset_index(drop=True)
    totResultsDf.to_csv(f'{csvFileName}.csv', index=False, header=True)
    print('lasted',ti()-t0,'s')
#%% 

#%% 

#%% 

#%% 

#%% 

