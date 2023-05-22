#%% imports
import os
base_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_folder)
import itertools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor#jjj
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from utils import q,ti
from envVarsPreprocess import envVars
import traceback
#%%
class trainTestXY:
    def __init__(self, xTrain, xTest, yTrain, yTest, xScaler, yScaler):
        self.xTrain = xTrain
        self.xTest = xTest
        self.yTrain = yTrain
        self.yTest = yTest
        self.xScaler = xScaler
        self.yScaler = yScaler


class predictedY:
    def __init__(self, name, prediction, trueValue):
        self.name = name
        self.prediction = prediction
        self.trueValue = trueValue

    def getScores(self, scoringDictionary):
        scores = {}
        for sdk, sdv in scoringDictionary.items():
            scores[self.name + sdk.capitalize()] = sdv(self.trueValue, self.prediction)
        return scores


class modelEvaluator:
    def __init__(self, name, modelFunc, hyperParamRanges, crossValidationNum):
        self.name = name
        self.modelFunc = modelFunc
        self.hyperParamRanges = hyperParamRanges
        self.crossValidationNum = crossValidationNum
        self.crossVal = self.crossValidationNum >= 2
        self.stratifiedKFold = StratifiedKFold(n_splits=crossValidationNum) if crossValidationNum>1 else None
        self.scoring = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'rocAuc': roc_auc_score,
            'cohenKappaScore': cohen_kappa_score
        }
        self.colsOrder = ['model', 'Parameter Set', 'Fold', 'testAccuracy', 'testPrecision', 'testRecall', 
            'testF1', 'testRocauc', 'testCohenkappascore']
        
        if self.crossVal:
            self.colsOrder.extend(['trainAccuracy', 'trainPrecision', 'trainRecall', 'trainF1', 'trainRocauc', 'trainCohenkappascore'])

    def fitModelAndGetResults(self, data,totResultsDf=None, parallel=True, paramCheckMode=envVars['paramCheckMode']):
        print(f'started fitting {self.name}')
        if paramCheckMode:
            if self.hyperParamRanges:
                for key, value in self.hyperParamRanges.items():
                    if any(isinstance(item, (int, float)) for item in value):
                        self.hyperParamRanges[key] = [item for item in value if isinstance(item, (int, float))][:1]
            
            if self.crossVal:
                self.crossValidationNum = 2
                self.stratifiedKFold = StratifiedKFold(n_splits=self.crossValidationNum) if self.crossValidationNum>1 else None        
        
        resultsDf = pd.DataFrame()
        inputArgs=[]
        
        #kkk separate the hyperParamRanges, totResultsDf.empty, parallel to their own funcs
        if self.hyperParamRanges:
            paramCombinations = list(itertools.product(*self.hyperParamRanges.values()))
            for params in paramCombinations:
                hyperparameters = dict(zip(self.hyperParamRanges.keys(), params))
                model = self.modelFunc(**hyperparameters)
                
                if self.crossVal:
                    for fold, (trainIndex, testIndex) in enumerate(self.stratifiedKFold.split(data.xTrain, data.yTrain)):
                        inputArgs.append([fold+1, data, model, hyperparameters, trainIndex, testIndex])
                else:
                    inputArgs.append(['noCrossVal', data, model, hyperparameters, list(range(len(data.xTrain))), []])
        else:
            model = self.modelFunc()
            if self.crossVal:
                for fold, (trainIndex, testIndex) in enumerate(self.stratifiedKFold.split(data.xTrain, data.yTrain)):
                    inputArgs.append([fold+1, data, model, '', trainIndex, testIndex])
            else:
                inputArgs.append(['noCrossVal', data, model, '', list(range(len(data.xTrain))), []])
        
        # check if the inputArgs are not in the totResultsDf
        if not totResultsDf.empty:
            for irI in range(len(inputArgs)-1,-1,-1):
                ir=inputArgs[irI]
                if not totResultsDf[(totResultsDf['model'] == self.name) & (totResultsDf['Parameter Set'] == str(ir[3]))
                    & (totResultsDf['Fold'] == ir[0])].empty:
                    inputArgs.remove(ir)
                    
        if parallel:
            with ProcessPoolExecutor(max_workers=max(multiprocessing.cpu_count() - 4, 1)) as executor:
                try:
                    resultRows = list(executor.map(self.processFold, *zip(*inputArgs)))
                    resultsDf = pd.concat([resultsDf, *resultRows]).reset_index(drop=True)
                except Exception as e:
                    print('ThreadPoolExecutor errrr', e)
        else:
            for ir in inputArgs:
                resultRows = self.processFold(*ir)
                resultsDf = pd.concat([resultsDf, resultRows]).reset_index(drop=True)
        return resultsDf
    def processFold(self, fold, data, model, hyperparameters, trainIndex, testIndex):
        try:
            q('processFold',fold, self.name, hyperparameters,ti(),filewrite=True)
            foldXTrain, foldXTest = data.xTrain[trainIndex], data.xTrain[testIndex]
            foldYTrain, foldYTest = data.yTrain[trainIndex], data.yTrain[testIndex]
    
            try:
                fitted_model = model.fit(foldXTrain, foldYTrain.ravel())
            except Exception as e:
                q('fitted_model Errrrrrr',fold, self.name, hyperparameters,'time:',ti(),'\nerrrr:',e,'\ntraceback:',traceback.format_exc(),'\nfoldXTrain:',type(foldXTrain),foldXTrain,'\nfoldYTrain.ravel():',type(foldYTrain.ravel()),foldYTrain.ravel(),'\n',filewrite=True,filename='fitted_modelErrrr')
            scores = {'model': self.name, 'Parameter Set': str(hyperparameters), 'Fold': fold}
    
            predicteds = [predictedY('test', fitted_model.predict(data.xTest), data.yTest)]
            if self.crossVal:
                predicteds.append(predictedY('train', fitted_model.predict(foldXTest), foldYTest))
            
            for predicted in predicteds:
                scores.update(predicted.getScores(self.scoring))
            
            dfRow = pd.DataFrame(scores, index=[0])[self.colsOrder]
            return dfRow
        except Exception as e:
            q('processFold Errrrrrr',fold, self.name, hyperparameters,'time:',ti(),'\nerrrr:',e,'\ntraceback:',traceback.format_exc(),'\n',filewrite=True,filename='processFoldErrr')
            return pd.DataFrame()