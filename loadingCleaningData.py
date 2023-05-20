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
#%% sec: handle categorical cols
#%% change True/False cols to 1/0
def trueFalseColsTo1_0(df):
    for col in df.columns:
        if df[col].dtype == bool:  # Check if the column is boolean
            df[col] = df[col].astype(int)  # Convert boolean to int
        else:
            unique_values = df[col].unique()
            if len(unique_values) == 2 and set(unique_values) == {True, False}:
                df[col] = df[col].astype(bool).astype(int)
    return df
telcoChurn=trueFalseColsTo1_0(telcoChurn)
#%% change yes/no cols to 1/0
def yesNoColsTo1_0(df):
    for col in df.columns:
        if df[col].dtype == object:  # Check if the column is of object type (string)
            unique_values = df[col].str.lower().str.strip().unique()
            if set(unique_values) == {'yes', 'no'}:
                df[col] = df[col].str.lower().str.strip().map({'yes': 1, 'no': 0})
    return df
telcoChurn=yesNoColsTo1_0(telcoChurn)
#%% handle categorical cols 
'''
#ccc
not considering the numerical cols we have categorical columns
they have some options like gender('male','female') or some of them have more than 2 options
for 2 option categorical cols like gender we do 'gender_female' and for females we put 1 and for males we do 0
for more than 2 option categorical cols like 'PaymentMethod' which has options of 'Bank transfer (automatic)','Credit card (automatic)','Electronic check' and 'Mailed check' we 'PaymentMethod_Bank transfer (automatic)','PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check' and 'PaymentMethod_Mailed check'. so for i.e. if the option in 'PaymentMethod' is 'Electronic check' we put 1 for that row and for other new columns we put 0
'''
#%% find categorical cols 
categoricalCols=telcoChurn.select_dtypes(exclude=['number']).columns
# Find categorical columns with 2 unique values
categoricalCols2unique = [col for col in categoricalCols if telcoChurn[col].nunique() == 2]

telcoChurnWithCategoricalCols=telcoChurn.loc[:, categoricalCols]
telcoChurnWithCategoricalCols=pd.get_dummies(telcoChurnWithCategoricalCols)

telcoChurnNonCategoricalCols=telcoChurn.loc[:,list(set(telcoChurn.columns) - set(categoricalCols))]
#%%
"#ccc here we remove 'gender_Male' from 'telcoChurnWithCategoricalCols' because gender has only 2 categories"
def delete2ndOptionFor_CategoricalCols2unique_From_telcoChurnWithCategoricalCols(categoricalCols2unique,mainDf,dfWithCategoricalCols):
    for cc2u in categoricalCols2unique:
        _2ndOption=mainDf[cc2u].unique()[1]
        dfWithCategoricalCols = dfWithCategoricalCols.drop(f'{cc2u}_{_2ndOption}', axis=1)
    return dfWithCategoricalCols
telcoChurnWithCategoricalCols=delete2ndOptionFor_CategoricalCols2unique_From_telcoChurnWithCategoricalCols(categoricalCols2unique,telcoChurn,telcoChurnWithCategoricalCols)
telcoChurnWithCategoricalCols=trueFalseColsTo1_0(telcoChurnWithCategoricalCols)
#%% put the telcoChurn cols back together
telcoChurn=pd.concat([telcoChurnNonCategoricalCols, telcoChurnWithCategoricalCols], axis=1)
telcoChurn = telcoChurn.dropna()
del telcoChurnWithCategoricalCols,telcoChurnNonCategoricalCols,categoricalCols2unique,categoricalCols
#%% sec: outliers
#%% describe numericCols
numericCols=[col for col in telcoChurn.columns if telcoChurn[col].nunique() > 2]
print('numericCols description:',telcoChurn[numericCols].describe())
#%% plot boxplots for outliers
'boxplots show the quartiles; dots shown in plots are outliers but they differ from outliers that we detect'
import matplotlib.pyplot as plt
import seaborn as sns
for nc in numericCols:
    sns.boxplot(x ='Churn', y = nc, data = telcoChurn)
    plt.title(nc)
    plt.show()
#%% outliers Checker with Inter Quartile Range (IQR)
#kkk I may add test driven to check these steps
from collections import Counter
outlierList = []
criticalOutlierColsForARow=1
for column in numericCols:
    # 1st quartile (25%)
    Q1 = np.percentile(telcoChurn[column], 25)
    # 3rd quartile (75%)
    Q3 = np.percentile(telcoChurn[column],75)
    # Interquartile range (IQR)
    IQR = Q3 - Q1
    # outlier step
    outlierStep = 1.5 * IQR
    # Determining a list of indices of outliers
    outlierListColumn = telcoChurn[(telcoChurn[column] < Q1 - outlierStep) | (telcoChurn[column] > Q3 + outlierStep )].index
    # appending the list of outliers 
    outlierList.extend(outlierListColumn)
    
# selecting observations containing more than x outliers
outlierList = Counter(outlierList)
multipleOutliers = list( k for k, v in outlierList.items() if v > criticalOutlierColsForARow )#ccc if only the row has more than n outliers would be detected as outLier
telcoChurn=telcoChurn.drop(multipleOutliers).reset_index(drop=True)
'this dataset didnt have outliers if it had we had shown the boxplots once more'
#%% sec: train_test_split
from sklearn.model_selection import train_test_split
xData = telcoChurn.drop('Churn',axis=1).values
yData = telcoChurn.Churn.values
# spliting the data into test and train
xTrain, xTest , yTrain, yTest = train_test_split(xData, yData , test_size=0.2, random_state=0)
#%% sec: imbalance
sns.countplot(x = 'Churn', data = telcoChurn)
plt.show()
'we see the no churn has more data than customers who churn'
print('no churn ratio:',len(telcoChurn[telcoChurn['Churn']==0])/len(telcoChurn))
"so on this data with just predicting 'no' we would get 73.5% correct answers!!"
#%% upsampling
from imblearn.over_sampling import SMOTE
doUpSampling=False
if doUpSampling:
    print('Before upsampling count of label 0 {}'.format(sum(yTrain==0)))
    print('Before upsampling count of label 1 {}'.format(sum(yTrain==1)))
    # Minority Over Sampling Technique
    sm = SMOTE(sampling_strategy = 1, random_state=1)
    xTrainSampled, yTrainSampled= sm.fit_resample(xTrain, yTrain.ravel())
                                             
    print('After upsampling count of label 0 {}'.format(sum(yTrainSampled==0)))
    print('After upsampling count of label 1 {}'.format(sum(yTrainSampled==1)))
#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

