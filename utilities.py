import numpy                         as np
import timeit
import matplotlib.pyplot             as plt
import pandas                        as  pd 
import keras


from   keras.layers                  import TimeDistributed 
from   keras.layers.convolutional    import SeparableConv1D
from   keras.layers.convolutional    import Conv1D
from   keras.layers.convolutional    import MaxPooling1D
from   keras.models                  import Sequential
from   keras.layers                  import Dropout
from   keras.layers                  import Dense
from   keras.layers                  import LSTM
from   keras.layers                  import Flatten
from   keras.layers                  import InputLayer
from   keras.regularizers            import L1L2
from   BLAS                          import *
from   tabulate                      import tabulate
from   sklearn.model_selection       import TimeSeriesSplit
from   sklearn.preprocessing         import MinMaxScaler
from   keras.callbacks               import ModelCheckpoint 
from   kerastuner.tuners             import BayesianOptimization

# Parameters
Horizon       = 4
nFeatures     = 1
nSubsequences = 1
nTimeSteps    = Horizon

# Network parameters
epochs     = 50
batch_size = 150

# other setting
chk_location = '../Final_model/weights.base.hdf5'
coin = 'BTC'

def data_retriever(coin):
    """
    Depending on the coin type, reads the data from csv; splits them into train, 
    cross validation and test sets. Since we past 'horizon' timesteps of data, 
    some data massaging is taken care of at the split of Train - CV sets and CV - Test sets.
    Returns the processed data
    """
    
    #Reading the data from the file
    coin_data_1h_raw = pd.read_csv(r'../data_binance/{}_USD_1h.csv'.format(coin))
    coin_close_1h = coin_data_1h_raw.loc[:, 'open'].values
    dates = coin_data_1h_raw.loc[:, 'timestamp']
    coin_close_1h = coin_close_1h.astype('float32')
    coin_close_1h = coin_close_1h.reshape((coin_close_1h.shape[0], 1))

    # Forming Training, Validation and Test data
    split_id = int(np.floor(coin_close_1h.shape[0] * 0.7))
    split_id_2 = int(np.floor(coin_close_1h.shape[0] * 0.8))

    #Splitting the data into Train, CV and Test set
    TrainingData = coin_close_1h[:split_id, :]
    ValidationData = coin_close_1h[split_id:split_id_2, :]
    TestingData = coin_close_1h[split_id_2:, :]
    
    """
    Adding last horizon values from Train data to Validation and from Validation to Test data 
    for enabling the initial predictions on Validation and Test set
    """
    ValidationData = np.concatenate((TrainingData[-Horizon:], ValidationData), axis=0)
    TestingData = np.concatenate((ValidationData[-Horizon:], TestingData), axis=0)

    
    return (TrainingData, ValidationData, TestingData, dates)

def Train_Valid_Test_split(coin):
    """
    Depending on coin type, it first gets the data and its first differences.
    Replaces the zeros with the sign of the mean value of price difference.
    Returns the processed data.
    """
    
    # Receiving the processed data from the data_retriever function
    TrainingData, ValidationData, TestingData, dates = data_retriever(coin)


    #Train Dataset Processing
    # Creates sequences of prices for each time stamp as per horizon
    X, Y = create_dataset(TrainingData, Horizon)
    # Computing the First difference for the data
    trainX, trainY, trainY_raw = ConstructFirstDifferencesData(X, Y)

    # Cross Validation Set Dataset Processing
    X, Y = create_dataset(ValidationData, Horizon)
    validX, validY, validY_raw = ConstructFirstDifferencesData(X, Y)

    # Test Set Dataset Processing
    X, Y = create_dataset(TestingData, Horizon)
    testX, testY, testY_raw = ConstructFirstDifferencesData(X, Y)
    
    # TrainX or TestX shape is  is .. x Horizon - 1
    # TrainY or TestY shape is .. x 1

    ## Reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], nSubsequences, nTimeSteps, nFeatures))
    validX = np.reshape(validX, (validX.shape[0], nSubsequences, nTimeSteps, nFeatures))
    testX = np.reshape(testX, (testX.shape[0], nSubsequences, nTimeSteps, nFeatures))
    
    """
    If there are any zeros in the data, we substitute with the mean, 
    as this is a up/down classification problem
    """
    trainY[trainY[:] == 0] = np.sign(np.mean(trainY))
    validY[validY[:] == 0] = np.sign(np.mean(validY))
    testY[testY[:] == 0] = np.sign(np.mean(testY))

    return (trainX, trainY, validX, validY, testX, testY, dates, trainY_raw, validY_raw, testY_raw)


def model_CNN_LSTM():
    """
    Constructs CNN LSTM model with author specified architecture
    """
    model = Sequential()
    
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=2, padding='valid', activation='relu'), 
                              input_shape=(None, nTimeSteps, nFeatures))) 
    model.add(TimeDistributed(Dropout(0.05)))

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, padding='valid', activation='relu'), 
                              input_shape=(None, nTimeSteps, nFeatures)) )
    model.add(TimeDistributed(Dropout(0.05)))     
    
    model.add(TimeDistributed(Flatten()))
   
    model.add(LSTM(50, activation='relu', dropout = 0.2, kernel_regularizer = L1L2(l1=0.01, l2=0.01)))
    model.add(Dense(1, activation = 'tanh'))  #Final activation tanh function
    
    print("\nModel Architecture \n")
    model.summary()
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  metrics = ['accuracy'])
    
    return model

def model_CNN():
    """
    Constructs CNN model with hyperparameters same as that of CNN part of the author specified architecture
    """
    model2 = Sequential()
    
    model2.add(Conv1D(filters=32, kernel_size=2, padding='valid', activation='relu',input_shape=(nTimeSteps, nFeatures)))
    model2.add(Dropout(0.05))

    model2.add(Conv1D(filters=64, kernel_size=2, padding='valid', activation='relu'))

    model2.add(Dropout(0.05))    
    model2.add(Flatten())
    
    model2.add(Dense(1, activation = 'tanh'))  #Final activation tanh function
    
    model2.summary()
    model2.compile(loss='mean_absolute_error', optimizer='adam', metrics = ['accuracy'])
    print("\nModel Architecture \n")
   
    
    return model2

def model_LSTM():
    """
    Constructs LSTM model with hyperparameters same as that of LSTM part of the author specified architecture
    """
    model3 = Sequential()
  
    model3.add(LSTM(50, activation='relu', dropout = 0.2, kernel_regularizer = L1L2(l1=0.01, l2=0.01)))
    model3.add(Dense(1, activation = 'tanh'))  #Final activation tanh function
    
    print("\nModel Architecture \n")
    model3.compile(loss='mean_absolute_error', optimizer='adam', metrics = ['accuracy'])
    
    return model3

def model_constructor(coin, model_type = None, split = 0.7):
    """
    Receives the processed data with passed split argument from Train_Valid_Test_split for a given coin.
    Depending on the model type, it receives the compiled model from model_CNN or model_LSTM or model_CNN_LSTM()
    It returns the model and processed data.
    """
    
    trainX,trainY,validX,validY,testX,testY, trainY_raw, validY_raw, testY_raw = Train_Valid_Test_split(coin, split)
   
    if model_type == 'CNN':
        model = model_CNN()
        trainX = trainX.reshape(trainX.shape[0], nTimeSteps, nFeatures)
        validX = validX.reshape(validX.shape[0], nTimeSteps, nFeatures)
        testX = testX.reshape(testX.shape[0], nTimeSteps, nFeatures)
        
        
    elif model_type == 'LSTM':
        model = model_LSTM()
        trainX = trainX.reshape(trainX.shape[0], nTimeSteps, nFeatures)
        validX = validX.reshape(validX.shape[0], nTimeSteps, nFeatures)
        testX = testX.reshape(testX.shape[0], nTimeSteps, nFeatures)
        
    elif model_type == 'CNN_LSTM':
        model = model_CNN_LSTM()
        
    
    return model,trainX,trainY,validX,validY,testX,testY

def model_trainer(coin, model_type):
    """
    For a given coin, it receives the processed data and compiled model from model_constructor function.
    Then it fits while using cross-validation set evaluation and callbacks to save the best model. 
    It returns the raw predictions and prediction after applying sign function.
    """

    #Getting the model and processed data
    model, trainX, trainY, validX, validY, testX, testY, dates, trainY_raw,\
    validY_raw, testY_raw = model_constructor(coin, model_type)

    #Checkpointer to store the best model based on accuracy on CV set
    checkpointer = ModelCheckpoint(filepath='../First_model/weights.best.from_scratch.hdf5',
                                   monitor="val_accuracy",
                                   verbose=2,
                                   save_best_only=True)

    print("\nModel Training \n")

    score = model.fit(trainX, trainY,
                      validation_data = (validX, validY),
                      epochs = epochs,
                      batch_size = batch_size,
                      callbacks = [checkpointer],
                      verbose=2)
    model.summary()

    #Plotting the training plots
    plt.plot(score.history['loss'])
    plt.plot(score.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    #Testing predictions
    model.load_weights('../First_model/weights.best.from_scratch.hdf5')
    testPredict = np.sign(model.predict(testX))
    valid_predict = np.sign(model.predict(validX))
    trainPredict = np.sign(model.predict(trainX))


    return (trainPredict, valid_predict, testPredict, model, trainY,\
           validY, testY, dates, trainY_raw, validY_raw, testY_raw)

def model_trainer_threshold(coin, model_type, threshold):
    """
    For a given coin, it receives the processed data and compiled model from model_constructor.
    Then it fits while using cross-validation set evaluation and callbacks to save the best model
    Given we use threshold, after the predictions are computed, threshold is applied and only 
    confident predictions are used. Finally, it returns the raw predictions and prediction after 
    applying sign function.
    """
    
    # Getting the model and processed data
    model, trainX, trainY, validX, validY, testX, testY, dates, \
    trainY_raw, validY_raw, testY_raw = model_constructor(coin, model_type)

    # Checkpointer to store the best model based on accuracy on CV set
    checkpointer = ModelCheckpoint(filepath='../First_model/weights.best.from_scratch.hdf5',
                                   monitor="val_accuracy",
                                   verbose=2,
                                   save_best_only=True)

    print("\nModel Training \n")

    score = model.fit(trainX, trainY,
                      validation_data=(validX, validY),
                      epochs=epochs,
                      batch_size=batch_size,
                      callbacks=[checkpointer],
                      verbose=2)
    model.summary()

    # Plotting the training plots
    plt.plot(score.history['loss'])
    plt.plot(score.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    model.load_weights('../First_model/weights.best.from_scratch.hdf5')

    tp = model.predict(testX)
    testPredict = np.empty(tp.shape)
    testPredict[:] = np.NaN

    # Counting the predictions only when the model is confident enough
    for x in range(tp.shape[0]):
        if np.abs(tp[x]) >= threshold:
            testPredict[x] = np.sign(tp[x])
            testY[x] = testY[x]
            testY_raw[x] = testY_raw[x]
        else:
            testPredict[x] = np.NaN
            testY[x] = np.NaN
            testY_raw[x] = np.NaN

    # Removing the data points where the model does not make any prediction for correct evaluation
    testPredict_clean = testPredict[~ np.isnan(testPredict)]
    testY_clean = testY[~ np.isnan(testY)]
    testY_raw_clean = testY_raw[~ np.isnan(testY_raw)]

    # Performing the same process on cross validation data 
    vp = model.predict(validX)
    valid_predict = np.empty(vp.shape)
    valid_predict[:] = np.NaN

    for x in range(vp.shape[0]):
        if np.abs(vp[x]) >= threshold:
            valid_predict[x] = np.sign(vp[x])
            validY[x] = validY[x]
            validY_raw[x] = validY_raw[x]
        else:
            valid_predict[x] = np.NaN
            validY[x] = np.NaN
            validY_raw[x] = np.NaN

    valid_predict_clean = valid_predict[~ np.isnan(valid_predict)]
    validY_clean = validY[~ np.isnan(validY)]
    validY_raw_clean = validY_raw[~ np.isnan(validY_raw)]

    # Performing the same process on train data
    tnp = model.predict(trainX)
    trainPredict = np.empty(tnp.shape)
    trainPredict[:] = np.NaN

    for x in range(tnp.shape[0]):
        if np.abs(tnp[x]) >= threshold:
            trainPredict[x] = np.sign(tnp[x])
            trainY[x] = trainY[x]
            trainY_raw[x] = trainY_raw[x]
        else:
            trainPredict[x] = np.NaN
            trainY[x] = np.NaN
            trainY_raw[x] = np.NaN

    trainPredict_clean = trainPredict[~ np.isnan(trainPredict)]
    trainY_clean = trainY[~ np.isnan(trainY)]
    trainY_raw_clean = trainY_raw[~ np.isnan(trainY_raw)]

    return (trainPredict_clean, valid_predict_clean, testPredict_clean, model, trainY_clean, \
           validY_clean, testY_clean, dates, trainY_raw_clean, validY_raw_clean, testY_raw_clean)

def ClassificationEvaluation(testY, Pred, plots = False):
    """
    Computes the evaluation metrics like Confusion matrix,
    Accuracy, F1 & ROC curve elements. Returns these metrics.
    """
    
    N        = testY.shape[0]
   
    CM       = metrics.confusion_matrix(testY, Pred)
    Accuracy = metrics.accuracy_score(testY, Pred)
    F1       = metrics.f1_score(testY, Pred)
    
    fpr, tpr, thresholds = metrics.roc_curve(testY, Pred, drop_intermediate = False)
    
    #Area under ROC curve
    AUC = metrics.auc(fpr, tpr)
    if (plots):
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                     label='ROC (AUC = %0.4f)' % AUC)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()
 
    return (CM, Accuracy, F1, AUC, tpr, fpr)

def results_plotter(coin, model_type, threshold_switch=False, threshold=0.667):
    """
    Gets the predictions from the model_trainer/ model_trainer_threshold functions, 
    plots the predictions distribution, computes
    the evaluation metrics (Accuracy, F1 score and confusion matrix) and prints them 
    """
    
   #If threshold switch False, resort to default model; else you one with threshold for predictions 
    if threshold_switch:
        trainPredict, valid_predict, testPredict, model, trainY, validY, testY, dates, \
        trainY_raw, validY_raw, testY_raw = model_trainer_threshold(coin, model_type, threshold)
    else:
        trainPredict, valid_predict, testPredict, model, trainY, validY, testY, dates, \
        trainY_raw, validY_raw, testY_raw = model_trainer(coin, model_type)

    fig, axs = plt.subplots(1)
    fig.suptitle('\nWeights Distribution for {} Price movement Prediction'.format(coin))
    n, bins, patches = axs.hist(testPredict, 10, density=True, facecolor='b', alpha=0.5)
    axs.set_ylabel('Predicted Sign')
    axs.set_xlabel('Direction')

    axes = plt.gca()

    plt.grid(True, alpha=0.5)
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.show()

    # Results Analysis
    print("\n Results Analysis \n")
    CM, Accuracy, F1, AUC, tpr, fpr = ClassificationEvaluation(testY, testPredict)
    print(tabulate(pd.DataFrame([Accuracy, AUC, F1], index=['Accuracy', 'F1 Score', 'Area Under ROC']),
                   headers=['Metric', 'Values'], tablefmt='psql'))

    # Check confusion matrix (by Livieris)
    print('\n Confusion matrix')
    print(tabulate(pd.DataFrame(CM, index=['True Negative', 'True Positive']),
                   headers=['Pred. Negative', 'Pred Positive'], tablefmt='psql'))

    print('\n')
    return (trainPredict, valid_predict, testPredict, trainY, validY, testY, dates,
            trainY_raw, validY_raw, testY_raw)


################################################################
################ FUNCTION FOR More Layers  #####################
################################################################

def model_CNN_LSTM_layers():
    model = Sequential()
    
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=2, padding='valid', activation='relu'), 
                              input_shape=(None, nTimeSteps, nFeatures))) 
    model.add(TimeDistributed(Dropout(0.05)))

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, padding='valid', activation='relu'), 
                              input_shape=(None, nTimeSteps, nFeatures)) )
    model.add(TimeDistributed(Dropout(0.05)))     
    
    model.add(TimeDistributed(Flatten()))
   
    model.add(LSTM(50, return_sequences=True, activation='relu', dropout = 0.2, 
                   kernel_regularizer = L1L2(l1=0.01, l2=0.01)))
    model.add(LSTM(50, return_sequences=True, activation='relu', dropout = 0.2, 
                   kernel_regularizer = L1L2(l1=0.01, l2=0.01)))
    model.add(LSTM(30))
    model.add(Dense(1, activation = 'tanh'))  #Final activation tanh function
    
    print("\nModel Architecture \n")
    model.summary()
    plot_model(model,show_shapes=True,to_file='Model_LSTM3.png')
    model.compile(loss='mean_absolute_error', 
                  optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  metrics = ['accuracy']) 
    return model

def model_constructor_layers(coin, split = 0.7):
    
    trainX,trainY,validX,validY,testX,testY, trainY_raw, validY_raw, testY_raw = Train_Valid_Test_split(coin, split)
   
    model = model_CNN_LSTM_layers()
        
    
    return model,trainX,trainY,validX,validY,testX,testY

def model_trainer_layers(coin):

    model, trainX, trainY, validX, validY, testX, testY, dates, trainY_raw,\
    validY_raw, testY_raw = model_constructor_layers(coin)

    #Checkpointer to store the best model based on accuracy on CV set
    checkpointer = ModelCheckpoint(filepath='../First_model/weights.best.layers.hdf5',
                                   monitor="val_accuracy",
                                   verbose=2,
                                   save_best_only=True)

    print("\nModel Training \n")

    score = model.fit(trainX, trainY,
                      validation_data = (validX, validY),
                      epochs = epochs,
                      batch_size = batch_size,
                      callbacks = [checkpointer],
                      verbose=2)
    model.summary()

    #Plotting the training plots
    plt.plot(score.history['loss'])
    plt.plot(score.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    #Testing predictions
    model.load_weights('../First_model/weights.best.from_scratch.hdf5')
    testPredict = np.sign(model.predict(testX))
    valid_predict = np.sign(model.predict(validX))
    trainPredict = np.sign(model.predict(trainX))


    return (trainPredict, valid_predict, testPredict, model, trainY,\
           validY, testY, dates, trainY_raw, validY_raw, testY_raw)

################################################################
################ FUNCTION FOR Hyperparameter Tuning ############
################################################################

def model_CNN_LSTM_bayes(hp):
    model = Sequential()
    
    model.add(TimeDistributed(Conv1D(filters=hp.Choice('num_filters',
                                                        values=[32, 64],
                                                        default=64), 
                                     kernel_size=2, padding='valid', 
                                     activation=hp.Choice('dense_activation',
                                                          values=['relu', 'tanh', 'sigmoid'],
                                                          default='relu')), 
                                     input_shape=(None, nTimeSteps, nFeatures))) 
    model.add(TimeDistributed(Dropout(0.05)))

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, padding='valid', 
                                     activation='relu'), 
                              input_shape=(None, nTimeSteps, nFeatures)) )
    model.add(TimeDistributed(Dropout(0.05)))     
    
    model.add(TimeDistributed(Flatten()))
   
    model.add(LSTM(50, activation='relu', dropout = 0.2, kernel_regularizer = L1L2(l1=0.01, l2=0.01)))
    model.add(Dense(1, activation = 'tanh'))  #Final activation tanh function
    
    print("\nModel Architecture \n")
    model.summary()
    model.compile(loss='mean_absolute_error', 
                  optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                            values=[1e-2, 1e-3, 1e-4])), 
                  metrics = ['accuracy'])
    return model
def results_plotter_bayes(model):
    fig, axs = plt.subplots(1)
    fig.suptitle('\nWeights Distribution for {} Price movement Prediction'.format(coin))
    n, bins, patches = axs.hist(testPredict, 10, density=True, facecolor='b', alpha=0.5)
    axs.set_ylabel('Predicted Sign')
    axs.set_xlabel('Direction')

    axes = plt.gca()
    
    plt.grid(True, alpha=0.5)
    plt.rcParams["figure.figsize"] = (10,7)
    plt.show()
    
    #Results Analysis
    print("\n Results Analysis \n")
    CM, Accuracy, F1, AUC,tpr, fpr = ClassificationEvaluation(testY, testPredict)
    print(tabulate(pd.DataFrame([Accuracy,AUC,F1], 
                                index = ['Accuracy', 'F1 Score', 'Area Under ROC']), 
                                headers = ['Metric', 'Values'], tablefmt = 'psql'))
          
    # Check confusion matrix (by Livieris)
    print('\n Confusion matrix')
    print(tabulate(pd.DataFrame(CM, index =['True Negative', 'True Positive']),
                   headers= ['Pred. Negative','Pred Positive'],tablefmt='psql'))

    print('\n')
    
    
################################################################
################ FUNCTION FOR MULTI-FEATURES ###################
################################################################
    
def data_retriever_multi(coin,cols_level = None, cols_diff = None):
    
    coin_raw = pd.read_csv(r'../data_binance/{}_USD_1h.csv'.format(coin),index_col=0)
    other_raw = pd.read_csv(r'../all_features_combine.csv',index_col=0)
    
    dates = coin_raw.index
    coin_raw = coin_raw.astype('float32')
    # we take the 1st Diff of our BTC open prices
    coin_raw = coin_raw['open'].diff()
    
     # we take the 1st Diff for all non stationnary variable
    if cols_diff != None:
        firstDiff_data = other_raw[cols_diff]
        firstDiff_data = firstDiff_data.diff()
    
    # we merge everything and we drop na
    if ((cols_level == None) and (cols_diff == None)):
        data = np.expand_dims(coin_raw, axis=0)
    elif cols_level == None:
        data = pd.concat([firstDiff_data, coin_raw],axis=1)
        data = data.dropna()
    else:
        data = pd.concat([firstDiff_data, other_raw[cols_level], coin_raw],axis=1)
        data = data.dropna()
    
    
    data = np.array(data)
    
    # Forming Training, Validation and Test data
    split_id = int(np.floor(data.shape[0] * 0.7))
    split_id_2 = int(np.floor(data.shape[0] * 0.8))

    TrainingData = data[:split_id,:]
    ValidationData = data[split_id:split_id_2,:]
    TestingData  = data[split_id_2:,:]
    
    # Adding last horizon values from Train data to Validation and from Validation to Test 
    # data for enabling the intial predictions on Validaiton and Test set
    ValidationData  = np.concatenate((TrainingData[-Horizon:], ValidationData), axis=0)  
    TestingData  = np.concatenate((ValidationData[-Horizon:], TestingData), axis=0)  
    #print(TrainingData.shape[0] + ValidationData.shape[0] + TestingData.shape[0], dates.shape[0] ) 
    #2x Horizons data added to valid and testing data
    #print(ValidationData.shape)
    return (TrainingData, ValidationData,TestingData, dates) 

def Train_Valid_Test_split_multi(coin, features_lvl = None, features_diff = None): 
    TrainingData, ValidationData, TestingData, dates = data_retriever_multi('BTC', cols_level = ['RSI','Stochastic_Oscillator',
                                                                                           'Bollinger_Low','Bollinger_High'], 
                                                                      cols_diff = ['Volume_BTC','USDJPY', 'EURUSD', 'GBPUSD',
                                                                                   'XAUUSD', 'WTIUSD','VIX','Realize_Vol'])
    
    
    # Always put the target variable to the last column (at the right)
    target_var = nFeatures-1

    trainX, trainY = create_dataset(TrainingData, Horizon, target_var)
    validX, validY = create_dataset(ValidationData, Horizon, target_var)
    testX, testY = create_dataset(TestingData, Horizon, target_var)
    
    # In order to match Prakarsh's outputs of this function we will independantly compute trainY_raw, validY_raw, testY_raw 
    # which are simply the first difference of the BTC price (without being flagged by -1 or 1) 
    
        ## Reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    ## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    print(trainX.shape)
    trainX = np.reshape(trainX, (trainX.shape[0], nSubsequences, Horizon, nFeatures))
    validX = np.reshape(validX, (validX.shape[0], nSubsequences, Horizon, nFeatures))
    testX  = np.reshape(testX,  (testX.shape[0],  nSubsequences, Horizon, nFeatures))
    print(trainX.shape)
    trainY_raw = trainY
    validY_raw = validY
    testY_raw = testY
    
    
    ## To remplace 0 values: 2 options:
    
#     1.
    trainY = np.sign(trainY)
    trainY[trainY==0] = -1
    validY =np.sign(validY)
    validY[validY==0] = -1
    testY =np.sign(testY)
    testY[testY==0] = -1
    
#     2.
#     trainY[trainY[:] == 0] = np.sign(np.mean(trainY))
#     validY[validY[:] == 0] = np.sign(np.mean(validY))
#     testY[testY[:] == 0] = np.sign(np.mean(testY))
    
    
    #print(trainX[0],trainY_raw[0])
    #print(validX[0],  validY_raw[0])
    #print(testX[0],testY_raw[0])
    return (trainX,trainY,validX,validY,testX,testY, dates, trainY_raw, validY_raw, testY_raw)

def model_constructor_multi(coin, model_type = None):
    
    trainX,trainY,validX,validY,testX,testY, dates,trainY_raw, validY_raw, testY_raw = Train_Valid_Test_split_multi(coin)
   
    if model_type == 'CNN':
        model = model_CNN()
        trainX = trainX.reshape(trainX.shape[0], nTimeSteps, nFeatures)
        validX = validX.reshape(validX.shape[0], nTimeSteps, nFeatures)
        testX = testX.reshape(testX.shape[0], nTimeSteps, nFeatures)
        
        
    elif model_type == 'LSTM':
        model = model_LSTM()
        trainX = trainX.reshape(trainX.shape[0], nTimeSteps, nFeatures)
        validX = validX.reshape(validX.shape[0], nTimeSteps, nFeatures)
        testX = testX.reshape(testX.shape[0], nTimeSteps, nFeatures)
        
    elif model_type == 'CNN_LSTM':
        model = model_CNN_LSTM()
        
    
    
    #print(trainX.shape)
    
    return model, trainX,trainY,validX,validY,testX,testY, dates,trainY_raw, validY_raw, testY_raw


def model_trainer_multi(coin, model_type):
    model,trainX,trainY,validX,validY,testX,testY, dates, trainY_raw, validY_raw, testY_raw = model_constructor_multi(coin, model_type)
    
    checkpointer = ModelCheckpoint(filepath     = '../Final_model/weights.best.from_scratch5.hdf5',
                               monitor          = "val_accuracy",
                               verbose          =  0,
                               save_best_only   = False)
    
    print("\nModel Training \n")
    #print(trainX.shape)
    score = model.fit(trainX, trainY,
                  validation_data = (validX, validY),
                  epochs          = epochs, 
                  batch_size      = batch_size, 
                  callbacks       = [checkpointer],
                  verbose         = 0)
    model.summary()
    
    plt.plot(score.history['loss'])
    plt.plot(score.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    
    model.load_weights('../Final_model/weights.best.from_scratch5.hdf5')
    testPredict = np.sign(model.predict(testX))
    valid_predict = np.sign(model.predict(validX))
    trainPredict = np.sign(model.predict(trainX))
    
    return trainPredict,valid_predict, testPredict, model, trainY, validY, testY, dates,trainY_raw, validY_raw, testY_raw

def model_trainer_threshold_multi(coin, model_type, threshold):
    """
    For a given coin, it receives the processed data and compiled model from model_constructor_multi.
    Then it fits while using cross-validation set evaluation and callbacks to save the best model
    Given we use threshold, after the predictions are computed, threshold is applied and only 
    confident predictions are used.
    """
    model,trainX,trainY,validX,validY,testX,testY, dates,trainY_raw, validY_raw, testY_raw= model_constructor_multi(coin, model_type)
    
    checkpointer = ModelCheckpoint(filepath         = '../Final_model/weights.best.from_scratch5.hdf5', 
                               monitor          = "val_accuracy", 
                               verbose          =  0, 
                               save_best_only   = True)

    #es = EarlyStopping(monitor='val_accuracy',baseline=0.35)
    print("\nModel Training \n")
    #print(trainX.shape)
    score = model.fit(trainX, trainY,
                  validation_data = (validX, validY),
                  epochs          = epochs, 
                  batch_size      = batch_size, 
                  callbacks       = [checkpointer],
                  verbose         = 0)
    model.summary()
    
    plt.plot(score.history['loss'])
    plt.plot(score.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim((0,3))
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    
    model.load_weights('../Final_model/weights.best.from_scratch5.hdf5')
    
    tp = model.predict(testX)
    testPredict = np.empty(tp.shape)
    testPredict[:] = np.NaN
    
    for x in range(tp.shape[0]):
        if np.abs(tp[x]) >= threshold:
            testPredict[x] = np.sign(tp[x])
            testY[x] = testY[x]
            testY_raw[x] = testY_raw[x]
        else:
            testPredict[x] = np.NaN
            testY[x] = np.NaN
            testY_raw[x] = np.NaN
            
    print(testPredict.shape, testY.shape, testY_raw.shape)
    
    testPredict_clean = testPredict[~ np.isnan(testPredict)]
    testY_clean = testY[~ np.isnan(testY)]
    testY_raw_clean = testY_raw[~ np.isnan(testY_raw)]
    print(testPredict_clean.shape, testY_clean.shape, testY_raw_clean.shape)
    
    vp = model.predict(validX)
    valid_predict = np.empty(vp.shape)
    valid_predict[:] = np.NaN
 
    for x in range(vp.shape[0]):
        if np.abs(vp[x]) >= threshold:
            valid_predict[x] = np.sign(vp[x])
            validY[x] = validY[x]
            validY_raw[x] = validY_raw[x]
        else:
            valid_predict[x] = np.NaN
            validY[x] = np.NaN
            validY_raw[x] = np.NaN
            
    print(valid_predict.shape, validY.shape, validY_raw.shape)
    valid_predict_clean = valid_predict[~ np.isnan(valid_predict)]
    validY_clean = validY[~ np.isnan(validY)]
    validY_raw_clean = validY_raw[~ np.isnan(validY_raw)]
    print(valid_predict.shape, validY.shape, validY_raw.shape)    
        
    tnp = model.predict(trainX)
    trainPredict = np.empty(tnp.shape)
    trainPredict[:] = np.NaN
    
    for x in range(tnp.shape[0]):
        if np.abs(tnp[x]) >= threshold:
            trainPredict[x] = np.sign(tnp[x])
            trainY[x] = trainY[x]
            trainY_raw[x] = trainY_raw[x]
        else:
            trainPredict[x] = np.NaN
            trainY[x] = np.NaN
            trainY_raw[x] = np.NaN
            
    print(trainPredict.shape, trainY.shape, trainY_raw.shape)  
    trainPredict_clean = trainPredict[~ np.isnan(trainPredict)]
    trainY_clean = trainY[~ np.isnan(trainY)]
    trainY_raw_clean = trainY_raw[~ np.isnan(trainY_raw)]
    print(trainPredict.shape, trainY.shape, trainY_raw.shape)  
    
    return trainPredict_clean,valid_predict_clean, testPredict_clean, model, trainY_clean, validY_clean, testY_clean, dates,trainY_raw_clean, validY_raw_clean, testY_raw_clean


def results_plotter_multi(coin, model_type, threshold_switch = False, threshold = 0.667):
    """
    Gets the predictions from the model_trainer/ model_trainer_threshold_multi functions, 
    plots the predictions distribution, computes
    the evaluation metrics (Accuracy, F1 score and confusion matrix) and prints them 
    """
    
    if threshold_switch:
        trainPredict,valid_predict, testPredict, model, trainY, validY, testY, dates,trainY_raw, validY_raw, testY_raw = model_trainer_threshold_multi(coin, model_type, threshold)
    else:
        trainPredict,valid_predict, testPredict, model, trainY, validY, testY, dates,trainY_raw, validY_raw, testY_raw = model_trainer(coin, model_type)
    
    fig, axs = plt.subplots(1)
    fig.suptitle('\nWeights Distribution for {} Price movement Prediction'.format(coin))
    n, bins, patches = axs.hist(testPredict, 10, density=True, facecolor='b', alpha=0.5)
    axs.set_ylabel('Predicted Sign')
    axs.set_xlabel('Direction')

    axes = plt.gca()
    
    plt.grid(True, alpha=0.5)
    plt.rcParams["figure.figsize"] = (10,7)
    plt.show()
    
    #Results Analysis
    print("\n Results Analysis \n")
    CM, Accuracy, F1, AUC,tpr, fpr = ClassificationEvaluation(testY, testPredict)
    print(tabulate(pd.DataFrame([Accuracy,AUC,F1], index = ['Accuracy', 'F1 Score', 'Area Under ROC']), headers = ['Metric', 'Values'], tablefmt = 'psql'))
          
    # Check confusion matrix (by Livieris)
    print('\n Confusion matrix')
    print(tabulate(pd.DataFrame(CM, index =['True Negative', 'True Positive']),headers= ['Pred. Negative','Pred Positive'],tablefmt='psql'))

    print('\n')
    return trainPredict,valid_predict, testPredict,trainY, validY, testY, dates,trainY_raw, validY_raw, testY_raw


