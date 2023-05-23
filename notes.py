'''
thoughts:
    maybe change it to ipynb
'''

'''
the done steps would have #done before them
steps to do:
    machine learning approaches:
        do data cleaning:
            #done add 'tenure' meaning
            #done Dropping duplicated
            #done drop missing values
            #done change categorical to multiple cols
            #done handle outliers;plot before and after plot for numerical cols outlier
            #done show input data imbalances
            #done 'optional default not to do oversampling' for imbalances
            #notNow add comment 'for other imbalances and outlier options'
            think about other methods may needed
            
        
        EDA(exploratory data analysis):
            #ccc not a necessity, I have more important things to do
            'Correlation Analysis' and 'Heatmap'(maybe useful, I may do it after all steps)
        
        do first model:
            #done Hyperparameter Tuning
            #done 5fold Cross-validation
            #asDone I had to do the gridSearch manually in order to get all scores for all models; this couldnt be done with grid_search of sklearn
            #done handle without hyperparam modelEvaluation
            #done add an option if the params are in some fileSaved dont do the function
            #done add scaler
            #done add option for not using cross validation
            #done add env variables
            #done add other models like lgbm+hyperparams
            #done do minimal model for hyperparams check and revise hyperparams
            #done check if some model has lots of errors
            add confusion matrices for best algorithm
            add drawing of AUC-ROC Curve for best algorithm
            add comments
            separate the modelEval for a single model and (xFold data) now its named fitModelAndGetResults but rename it and make it better
            #kkk add subSteps later
        
        explain output data:
            with confusion matrices,roc and auc plots, accuracy,recall, f1score,Cohen kappa score,... or top features affecting the model    
            add explaination on 'what metrics better to be chosen'
        
        apply clean Architecture and code principles and TDD and DDD as possible:
            #kkk add subSteps later as needed
        
        do ensemble model:
            #kkk add subSteps
        
        take a look what further I can imporve:
            #kkk add the list of imporvements here
        

    deep learning approaches(should be done after 'ml approaches'):
        try vae classification
        try lstm
        try monte carlo dropout
'''