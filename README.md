# project utility

this project is for binary classification. the `deep learning` and `machine learning` methods can be applied

## deep learning

with `dropout ensemble model` u can use sudoProbabilistic approximation of binary classification with regular artificial neural networks

## machine learning

the common packages mostly just give best model for grid search. but this code can apply `cross fold validation` for all model while doing the `grid search`, also calculate 'train' and 'test' scores for multiple metrics. note if u are looking for a strategy which needs `test` data scores, this project can be helpful as most packages don't provide such.

has also feature of cpu parallelization for faster computations. the code structure has tried to be more memory efficient but can be improved a bit also.

this code mainly aims https://www.kaggle.com/blastchar/telco-customer-churn dataset to build a binary classification model to predict whether a customer will churn (i.e., discontinue their subscription). even though the data preprocessing part is written in general form, but its not recommended to use to clean ur data without checking the steps suitable for ur data.

# how to run + some explanations

## machine learning

to run the code you need first define `envVars` in `envVarsPreprocess.py`, then define the ML models and their hyperparameters you want to do gridSearch as part of ur hyperparameters tuning in `modelConfigs.py`. note u can also have no params specified, its recommended to first put `paramCheckMode` on to prevent params conflit with sklearn. then in `getBestModelsAndVotingMLClassifierForEachMetric.py` u would be asked to train the ML classifiers if they are not, afterward the results of best models with least acceptable score for multiple metrics('accuracy','precision','recall','f1','rocAuc','cohenKappaScore') would be available, beside the voting Classifier which classifies on they majority votes of the best models.

## deep learning

### dropout ensemble model

run the codes in `trainAndEvalDropoutNet.py`, note u can aso modify the architecture by changing `__init__` in `dropoutNet` class in `dropoutNetModule.py`
