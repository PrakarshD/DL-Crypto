# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 02:09:17 2019

@author: poseidon

8/31 Update by Yvonne Zhu
- create_dataset adds the parameter target_var to 
    allow multiple features' target designation

"""
import numpy                          as np
import pandas
import math
import matplotlib.pyplot              as plt
# import only system from os 
from   os                             import system
from   os                             import name 

from   sklearn.metrics                import mean_squared_error
from   sklearn.metrics                import mean_absolute_error
from   sklearn.metrics                import matthews_corrcoef
from   sklearn                        import metrics
from   statsmodels.stats.diagnostic   import acorr_ljungbox
from   statsmodels.graphics.tsaplots  import plot_pacf
from   statsmodels.graphics.tsaplots  import plot_acf
   





# Function: convert an array of values into a dataset matrix
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def create_dataset(dataset, look_back=1, target_var = 0):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        if target_var!=0:
            a = dataset[i:(i+look_back), :]
        else:
            a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, target_var])
    return np.array(dataX), np.array(dataY)





def getLogData(filename):
    dataframe = pandas.read_csv(filename, usecols=[1], sep=',')
    
    data = dataframe.values
    data = data.astype('float32')
    
    # return 'Log values'    
    return (np.log(data))
    





def getData(filename):
    dataframe = pandas.read_csv(filename, usecols=[1], sep=',')
    
    data = dataframe.values
    data = data.astype('float32')
    
    # return 'values'
    return(data)


def ConstructFirstDifferencesData(X, Y):
    trainX = np.zeros( (X.shape[0], X.shape[1]-1) )
    trainY = np.zeros( (Y.shape[0], ) )
    
    for i in range( X.shape[0] ):
        trainY[i] = Y[i] - X[i][-1]
        
        for j in range(0, X.shape[1]-1):            
            trainX[i][j] = X[i][j+1] - X[i][j]
        
    return (trainX, trainY)




def ConstructFirstDifferencesData1(X, Y):
    trainX = np.zeros( (X.shape[0], X.shape[1]-1) )
    trainY = np.zeros( (Y.shape[0], ) )  
    trainY_raw = np.zeros( (Y.shape[0], ) )  
    for i in range( X.shape[0] ):
        trainY[i] = np.sign(Y[i] - X[i][-1])  #Sign of movement
        trainY_raw[i]  = Y[i] - X[i][-1]
        
        for j in range(0, X.shape[1]-1):            
            trainX[i][j] = X[i][j+1] - X[i][j]
        
    return (trainX, trainY, trainY_raw)





 
    
    
def EvaluateModel(testY, testPredict):
    MAE  = mean_absolute_error(testY, testPredict)
    RMSE = math.sqrt(mean_squared_error(testY, testPredict))
    MAPE = np.mean(np.abs((testY - testPredict) / testY)) * 100
    
    
    return (MAE, RMSE, MAPE)
    







def ClassificationEvaluation(testY,Pred):
    N = testY.shape[0]
    
   
    CM = metrics.confusion_matrix(testY, Pred)
    
    Accuracy = metrics.accuracy_score(testY, Pred)
    
    #McN      = abs(CM[0][1]-CM[1][0]) / math.sqrt(CM[0][1] + CM[1][0])
    #GM       = math.sqrt(CM[0][0] * CM[1][1])
    
    F1       = metrics.f1_score(testY, Pred)
    
    fpr, tpr, thresholds = metrics.roc_curve(testY, Pred, drop_intermediate = False)
    
    #Area under ROC curve
    AUC = metrics.auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % AUC)
    plt.legend(loc='lower right')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()
    """
    Sen      = CM[1][1] / (CM[1][1] + CM[1][0])
    Spe      = CM[0][0] / (CM[0][0] + CM[0][1])    

    
    if (sum(CM[:][1]) != 0):
        PPV      = CM[1][1] / (CM[1][1] + CM[0][1])
    else:
        PPV      = 0.0
        
    if ( (CM[0][0] + CM[1][0]) != 0):
        NPV      = CM[0][0] / (CM[0][0] + CM[1][0])
    else:
        NPV      = 0.0
    """
    
    return (CM, Accuracy, F1, AUC, tpr, fpr)






    
def AutoCorrelationResidual_Test(testY, testPredict):   
    # Auto-corretion check (by Stavroyiannis)
    
    #Prediction Error
    D = testY - testPredict[:,0]

    # ACF plot
    fig = plot_acf(D, lags=5)
    fig.set_dpi(100)
    fig.set_figheight(3.0)
    plt.show()           
        
#    # PCF plot
#    fig = plot_pacf(D, lags=5)
#    fig.set_dpi(100)
#    fig.set_figheight(3.0)
#    plt.show()           
        
    # Apply
    # 1. Ljung-Box Q-test
    # 2. Box-Pierce test    
    TestForResidualCorrelation(D) 
    
    
    
    
    
      
def TestForResidualCorrelation(D):

    lbvalue, pvalue, bpvalue, bppvalue = acorr_ljungbox(D, lags=[10], boxpierce=True)    

    print('\n\n')
    print('Ljung-Box Q-test    ***   p-value = %5.4f' % pvalue)
    print('H0: Auto correlations upto lags {} are 0'.format(10))
    if (pvalue < 0.05):
        print('H0 is REJECTED\n\n')
    else:
        print('Fail to reject H0\n\n')
    
    print('Box-Pierce test     ***   p-value = %5.4f' % bppvalue)
    if (bppvalue < 0.05):
        print('H0 is REJECTED\n\n')
    else:
        print('Fail to reject H0\n\n')
        
        







