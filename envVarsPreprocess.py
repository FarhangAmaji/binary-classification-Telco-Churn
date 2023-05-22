envVars={'crossValNum':2,
         'upSampling':False,
         'parallel':False}

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
