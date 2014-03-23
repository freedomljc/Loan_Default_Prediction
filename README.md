Loan_Default_Prediction
=======================

This is the Python Code for the submission to Kaggle's Loan Default Prediction by the ID "HelloWorld"

My best score on the private dataset is 0.44465, a little better than my current private LB score 0.44582, ranking 2 of 677. Using this script, you can yield similiar results with my best entry(score: 0.44465).
## Data preprocessing: 
    The training data is sorted by the time, and the test data is randomly orded. So in the validation 
    process, I first shuffle the training data randomly.
    Owing to lack of the feature description, It is hard to use the tradition method to predict LGD. In 
    my implemention, the operator +,-.*,/  between two features, and the operator (a-b) * c among three 
    features were used, these features were selected by computing the pearson corrlation with the loss.
## Model description:
    GBM classifier(traindata_1) -> guassian process regression
    GBM calssifier(traindata_2) -> svr, GBM regression
    Finally, the prediction results from guassian process regression, svr, GBM regression are blended 
    linearly.
    Otherwise, owing to the long tail distribution of loss, the log(loss) was used.
## Requirements:
    sklearn package, about 96G ram(gaussian process process spend too much memory).
## Instruction:
    Download the data from http://www.kaggle.com/c/loan-default-prediction/data to the working directory
    'loan_default_prediction'.
    Run the script predict.py
    
