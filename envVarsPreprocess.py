from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,cohen_kappa_score
envVars={'crossValNum':0, #0 and 1 are for noCrossVal
         'upSampling':True,#to cure imbalancing of data
         'parallel':True,#cpu parallelization for potentially get faster results
         'paramCheckMode':False#its recommended when u want to add new modelConfig
         }

if envVars["crossValNum"] < 2:
    envVars['cv']=False
else:
    envVars['cv']=True

envVars['csvFileName']='churnTotResults'
if envVars['cv']:
    envVars['csvFileName']+=str(envVars["crossValNum"])+'cv'
else:
    envVars['csvFileName']+='NoCv'
if not envVars['upSampling']:
    envVars['csvFileName']+='NoUpsampling'
envVars['csvFileName']=r'data\outputs'+'\\'+envVars['csvFileName']
envVars["metrics"]={
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'rocAuc': roc_auc_score,
    'cohenKappaScore': cohen_kappa_score
}