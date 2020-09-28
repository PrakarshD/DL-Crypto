# Bitcoin price prediction with Deep Learning 
#### This repository's work is inspired by ["A novel validation framework to enhance deep learning models in time-series forecasting" (Livieris et al.)](https://www.researchgate.net/publication/341755634_A_novel_validation_framework_to_enhance_deep_learning_models_in_time-series_forecasting) and ["Investigating the Problem of Cryptocurrency Price Prediction: A Deep Learning Approach" (Pintelas et al.)](https://link.springer.com/chapter/10.1007/978-3-030-49186-4_9)
#### Full report can be found [here](https://github.com/wingwingz/DL-Crypto/blob/master/Final_Report.pdf)
#### For best view and the ability to navigate via table of contents, please visit our [colab version here]( https://colab.research.google.com/drive/17_DSYw9d2MnJqLXqr09k2qvw3z0Xzv6j?usp=sharing)

## Original Work Description & Motivations
In "A novel validation framework to enhance deep learning models in time-series forecasting" (Livieris et al.), it was theoretical and empirically proved the proposed framework based on differencing can significantly improve the efficiency and reliability of any (deep learning) model. More analytically, it present why the series should be differenced in order to improve the performance of a deep learning model. 
<br>
To verify and advance the original authors' work, we present the following repository with more CNN and LSTM architectures, inclusion of external datasets, and various model improvement methods along with transaction cost analysis. 

## Datasets Folder
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

## Demo Results
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
