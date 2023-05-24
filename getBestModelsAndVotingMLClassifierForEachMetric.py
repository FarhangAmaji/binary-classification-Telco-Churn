#%% imports
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)

from allModelImports import *
from modelConfigs import allModelConfigs, trainTestXY_
from envVarsPreprocess import envVars
from utils import q, ti, inputTimeout
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore scikit-learn warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#%%
def runTrainMLClassifers():
    inpRes=inputTimeout('run train classifers?\n"anything would run but leave it empty if u dont want"',30)
    needToRunMLClassifiers=False
    if not inpRes:
        csvFileName=envVars['csvFileName']
        if os.path.exists(f'{csvFileName}.csv'):
            totResultsDf = pd.read_csv(f'{csvFileName}.csv')
            if totResultsDf.empty:
                needToRunMLClassifiers=True
        else:
            needToRunMLClassifiers=True
    else:
        needToRunMLClassifiers=True
    if needToRunMLClassifiers:
        from trainMlClassifiers import totResultsDf
    return totResultsDf
totResultsDf=runTrainMLClassifers()
#%% best model for each metric on test data
#kkk for each metric draw confusion matrix and roc
leastScoresOfEachMetric={'accuracy': .75,
'precision': .75,
'recall': .75,
'f1': .55,
'rocAuc': .65,
'cohenKappaScore': .45}
def getMetricsBestOfEachModel(wantedMetricName):
    assert wantedMetricName in envVars["metrics"].keys()
    metricScoreName='test' + wantedMetricName.capitalize()
    metricLeastScore=totResultsDf[totResultsDf[metricScoreName]>leastScoresOfEachMetric[wantedMetricName]]
    metricSortedDf = metricLeastScore.sort_values(by=metricScoreName, ascending=False)
    bestOfEachModelForMetricDf = metricSortedDf.drop_duplicates(subset='model', keep='first').reset_index(drop=True)
    return bestOfEachModelForMetricDf
def getModelFuncOfModelConfig(modelName):
    for m1 in allModelConfigs:
        if modelName==m1.name:
            return m1.modelFunc
    return False
def getAllBestEstimatorsForMetric(metricName):
    allBestEstimatorsForMetric=[]
    bestOfEachModelForMetricDf=getMetricsBestOfEachModel(metricName)
    for bmIndex in range(len(bestOfEachModelForMetricDf)):
        modelName=bestOfEachModelForMetricDf.loc[bmIndex,'model']
        parameterSet=bestOfEachModelForMetricDf.loc[bmIndex,'Parameter Set']
        if isinstance(parameterSet, str):
            parameterSet=eval(parameterSet)
        elif np.isnan(parameterSet):
            parameterSet={}
        modelFunc=getModelFuncOfModelConfig(modelName)
        if modelFunc:
            modelFuncWithHyperparams=modelFunc(**parameterSet)
            allBestEstimatorsForMetric.append((modelName,modelFuncWithHyperparams))
    return allBestEstimatorsForMetric, bestOfEachModelForMetricDf
#%% 
allVotingClassifiers={}
for metricName, metricFunc in envVars['metrics'].items():
    thisVotingClassifier={}
    print(metricName)
    allBestEstimatorsForMetric, bestOfEachModelForMetricDf=getAllBestEstimatorsForMetric(metricName)
    votingClassifier = VotingClassifier(estimators = allBestEstimatorsForMetric, voting ='hard')
    "#ccc voting hard means more than half of the estimators decide on sth"
    votingClassifier.fit(trainTestXY_.xTrain, trainTestXY_.yTrain.ravel())
    
    thisVotingClassifier['metric'] = metricName
    thisVotingClassifier['votingClassifierModel'] = votingClassifier
    
    yPred = votingClassifier.predict(trainTestXY_.xTest)
    scoreOfVotingClassifier=metricFunc(trainTestXY_.yTest, yPred)
    thisVotingClassifier['scoreOfVotingClassifier'] = scoreOfVotingClassifier
    
    metricScoreName='test' + metricName.capitalize()
    print(f'voting classifier of {metricName} is {scoreOfVotingClassifier:.3f} while best score was {bestOfEachModelForMetricDf.loc[0,metricScoreName]:.3f}')
    cm = confusion_matrix(trainTestXY_.yTest, yPred)
    
    # Create the ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Churn", "No Churn"])
    
    # Plot the confusion matrix with the desired color map
    ax = disp.plot(cmap=plt.cm.OrRd_r)

    # Add a title to the plot
    plt.title(f"Confusion Matrix - Voting Classifier for {metricName}")
    
    plt.show()
    
    allVotingClassifiers[metricName] = thisVotingClassifier

#%% 

#%% 

#%% 

