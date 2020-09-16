def price_pattern(update):

    '''
    Goal: generate BTC open price related factors and y-label 
    
    Added_feature description:
    first diff: first difference of the open price series
    true_flag: the true time moment which is the threshold
    flag_before: avoid the look-ahead issue, shift true_flag one moment 
    diff_threshold: calculate the price difference between current price and previous threshold
    target: y label -> up(+1) and down(-1)
    
    input: 
    - update: dataframe, raw dataset with BTC prices and other external factors 
    
    output: 
    - update: dataframe, updated dataset with added_features
    '''
    
    update['first_diff'] = update.Open.diff().fillna(0)
    update['true_flag'] = 0
    update['flag_before'] = 0
    threthold = update.loc[1, 'first_diff']
    update.loc[0, 'diff_threthold'] = 0
    update.loc[1, 'diff_threthold'] = update.Open.values[1] - update.Open.values[0] 
    threthold = update.Open.values[0]

    # find local max/min and price difference from last threthold 
    # also calculate the speed of changing prices
    
    for i in range(2, len(update)):
        prev_diff = update.loc[i-1]['first_diff']
        #update.loc[i, 'diff'] = update.close.values[i] - threthold
        current_diff = update.loc[i, 'first_diff']
        if (np.sign(current_diff) != np.sign(prev_diff)):
            #if (abs(current_diff) > 5):
                if (update.first_diff.values[i] < 0):
                    update.loc[i-1, 'true_flag'] = 1
                    update.loc[i,'flag_before'] = 1
                elif ((update.first_diff.values[i] > 0)):
                    update.loc[i-1, 'true_flag'] = -1
                    update.loc[i,'flag_before'] = -1
                threthold = update.loc[i-1, 'Open']
        update.loc[i, 'diff_threthold'] = update.Open.values[i] - threthold    
    
    # create the momentum factor
    update['pct_change_close'] = update.Open.pct_change().fillna(0)
    
    # y_label
    y=[]
    y.append('NA')
    for i in range(1,len(update)):
        if(update['Open'][i]>=update['Open'][i-1]):
                   y.append(1)
        else:
                   y.append(-1)
    update['target']=y
    
    return update

def ADF_test(df): 
    '''
    Goal: ADF test to time series
    
    input: 
    - df: series, univariate time series data 
    
    output:
    Calculate and print out the result (P-values, Critical value) of ADF test for unit root
    
    '''
    adf_result1 = sm.tsa.stattools.adfuller(df)
    print('Close price: ADF statistic (p-value): %0.3f (%0.3f)' % (adf_result1[0], adf_result1[1]),
                          '\n\tcritical values', adf_result1[4],'\n')
    

def standard_average(size, train_btc):
    '''
    Goal: 
    - check the performance of normal average prediction using BTC open price
    - prediction at t+1 is the average value of all the prices observed within a window of t to  t−N
    
    Inputs:
    - Size: integer, determine the window size
    - train_btc: dataframe, the raw dataset with open price of BTC
    
    Output:
    - prediction list
    - print out Mean-squared error of prediction
    - plot the actual Price series and predicted price series 
    '''
    
    window_size = size
    N = len(train_btc)
    std_avg_predictions = []
    #std_avg_x = []
    mse_errors = []

    for pred_idx in range(window_size,N):
        std_avg_predictions.append(np.mean(train_btc[pred_idx-window_size:pred_idx]['open']))
        mse_errors.append((std_avg_predictions[-1]-train_btc.loc[pred_idx]['open'])**2)
        #std_avg_x.append(date)

    print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))
    
    plt.figure(figsize = (18,9))
    plt.plot(range(train_btc.shape[0])[:500],train_btc.loc[:500-1]['open'],color='b',label='True')
    plt.plot(range(window_size,N)[:500],std_avg_predictions[:500],color='orange',label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Open price')
    plt.legend(fontsize=18)
    plt.title(f'Window Size = {size}')
    plt.show()