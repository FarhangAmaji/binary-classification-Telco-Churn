#%% change dir to currentDir
import os
baseFolder=os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
#%% load data
import pandas as pd
filePath = os.path.join(baseFolder, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
telcoChurn = pd.read_csv(filePath)
del baseFolder,filePath
#%% sec: correct type of cols + categorical cols
#!!!!
#%% drop customerIDcol
'#ccc customerID adds no value to classification'
telcoChurn = telcoChurn.drop('customerID', axis=1)
#%% TotalCharges to float
import numpy as np
print('dtypes:',telcoChurn.dtypes)
"#ccc apparently the type of 'TotalCharges' can be changed to float"
telcoChurn['TotalCharges'] = telcoChurn['TotalCharges'].replace(' ', np.nan)
telcoChurn['TotalCharges'] = telcoChurn['TotalCharges'].astype(float)
#%% dealing missing values
telcoChurn = telcoChurn.dropna()
#%% removing duplicates
telcoChurn = telcoChurn.drop_duplicates()
#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

