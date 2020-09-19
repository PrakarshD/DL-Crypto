# Bitcoin price prediction with Deep Learning 
#### Full report can be found [here] (230T2%20Final%20Report%20-%20DL%20for%20Crypto.pdf)
#### For best view and the ability to navigate via table of contents, please visit our [colab version here]( https://colab.research.google.com/drive/17_DSYw9d2MnJqLXqr09k2qvw3z0Xzv6j?usp=sharing)
## Datasets folder
1. 'BTC_USD_1h.csv': original Bitocin price data gathered from Bistamps 
2. 'all_features_combine.csv': combined dataset with Bitcoin prices and all external factors

Will used the all_features_combine.csv for further analysis.

## Analysis Functions
#### a) BLAS.py has the following functions that transform the raw data to model inputs:
* create_dataset
* getLogData
* ConstructFirstDifferencesData
* EvaluateModel
* ClassificationEvaluation
* AutoCorrelationResidual_Test
* TestForResidualCorrelation

#### b) feature_eng.py has the following functions which help in feature_engineering and EDA procedure:
* ADF_test
* price_pattern: calculated open price related features
* standard_average: simple slide window average prediction

#### c) trans_cost.py trained the model with transaction costs
* TC_modeller
* TC_plotter

#### d) utilities.py contains all the functions which help in CNN and LSTM model training, such as:
* Train_Valid_Test_split
* model_CNN_LSTM, model_CNN, model_LSTM
* model_constructor, model_trainer
* model_trainer_threshold: adding accuracy threshold 
* model_CNN_LSTM_bayes: Parameter Tunning

## Demo results
#### We created a Demo.ipynb to present our analysis process and sample results for all trained models with following sections

[Data Preparation and Feature Engineering](#first)

[Baseline - SVM](#second)

[DL Models](#third)
* [Base Model: CNN-LSTM with Open Price Data](#cnnlstm)
* [Base Model: CNN-LSTM with Open Price Data & Thresholds](#tsd)
* [Base Model: Hyperparameter Tuning](#ht)
* [Base Model: More Layers](#layers)
* [Base Model: Rolling Horizon](#rh)
* [Advanced Model - External Data](#ed)
