#%% imports
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)

from modelConfigs import allModelConfigs
from envVarsPreprocess import envVars
from utils import q,ti, inputTimeout
#%%
t0=ti()
inputTimeout(f'have u checked the envVars.\nenvVars={envVars}',30)
csvFileName=envVars['csvFileName']
if os.path.exists(f'{csvFileName}.csv'):
    totResultsDf = pd.read_csv(f'{csvFileName}.csv')
    #kkk I can check for the file lock(file being used by other programs) by resaving it
else:
    totResultsDf = pd.DataFrame()
#%%
modelsAndTheirInputArgsPool=[]
if __name__ == '__main__':
    for m1 in allModelConfigs:
        modelsAndTheirInputArgsPool.append(m1.getInputArgs(totResultsDf))
    print('start eval',ti()-t0,'s')
    if envVars['parallel']:
        with ProcessPoolExecutor(max_workers=max(multiprocessing.cpu_count() - 4, 1)) as executor:
            resultRows = []
            for mir in modelsAndTheirInputArgsPool:
                for ir in mir[1]:
                    future = executor.submit(mir[0].fitModelAndGetResults, *ir)
                    resultRows.append(future.result())
                totResultsDf = pd.concat([totResultsDf, *resultRows]).drop_duplicates().reset_index(drop=True)
                totResultsDf.to_csv(f'{csvFileName}.csv', index=False, header=True)
    else:
        for mir in modelsAndTheirInputArgsPool:
            for ir in mir[1]:
                resultRows = mir[0].fitModelAndGetResults(*ir)
                totResultsDf = pd.concat([totResultsDf, resultRows]).reset_index(drop=True)
                totResultsDf.to_csv(f'{csvFileName}.csv', index=False, header=True)
    print('lasted',ti()-t0,'s')
#%% 

#%% 

#%% 

#%% 

