def TC_plotter(is_bc, valid_bc, os_bc, is_ac, valid_ac, os_ac ,dates, step, switch):

    #Plotting function for the Pnl series with TC
    fig, axs = plt.subplots(1)
    fig.suptitle('\n {} Pnl Analysis with Transaction costs'.format('BTC'))

    axs.plot(np.arange(len(os_bc)) ,os_bc ,'tab:red', label = 'OS BC Pnl ')
    axs.plot(np.arange(len(valid_ac)) ,valid_bc ,'tab:blue', label = 'Cross Validation BC Pnl ')
    axs.plot(np.arange(len(is_ac)) ,is_bc, 'tab:orange', label = 'In Sample BC Pnl')

    axs.plot(np.arange(len(os_bc)) ,os_ac ,'tab:cyan', label = 'OS AC Pnl ')
    axs.plot(np.arange(len(valid_ac)) ,valid_ac ,'tab:green', label = 'Cross Validation AC Pnl ')
    axs.plot(np.arange(len(is_ac)) ,is_ac, 'tab:brown', label = 'In Sample AC Pnl')

    axs.xaxis.set_major_locator(plt.NullLocator())
    axs.set_ylabel('PNL')
    axs.set_xlabel('Time')

    axes = plt.gca()

    plt.grid(True, alpha=0.5)
    plt.rcParams["figure.figsize"] = (15 ,10)

    plt.legend()
    plt.show()
    
def TC_modeller(trainY_raw, validY_raw, testY_raw, train_predict, valid_predict, test_predict,
                dates, tc_coeff, book_size=1, switch=True):
    # Before Cost Analysis

    c_pnl_BC = 0.0  # Variable to store cumulative before cost PnL

    # Before Cost list variables
    is_pnl_series_BC = []
    valid_pnl_series_BC = []
    os_pnl_series_BC = []

    # Analysis for Train data
    for x in range(trainY.shape[0]):
        # Adding the PnL for a time stamp to the cumulative variable
        c_pnl_BC += float(book_size * (train_predict[x]) * trainY_raw[x])

        # Storing the variable to the lists
        is_pnl_series_BC.append(c_pnl_BC)
        valid_pnl_series_BC.append(c_pnl_BC)
        os_pnl_series_BC.append(c_pnl_BC)
        # print(is_pnl_series_BC)

    # Analysis for Cross Validation (CV) data
    for x in range(validY.shape[0]):
        c_pnl_BC += float(book_size * (valid_predict[x]) * validY_raw[x])

        valid_pnl_series_BC.append(c_pnl_BC)
        os_pnl_series_BC.append(c_pnl_BC)

    # Analysis for Test data
    for x in range(testY.shape[0]):
        c_pnl_BC += float(book_size * (test_predict[x]) * testY_raw[x])

        os_pnl_series_BC.append(c_pnl_BC)

    # After Cost Analysis

    # After Cost list variables
    c_pnl_AC = 0.0
    is_pnl_series_AC = []
    valid_pnl_series_AC = []
    os_pnl_series_AC = []

    # Transaction Cost coeff: How much $ one pays for 1 unit of coin on side of the trade
    tc_per_trade_per_coin = tc_coeff

    # After cost analysis on Train data
    for x in range(trainY.shape[0]):

        # Initial_Position_setup: requires for deduction of TC related to position formation
        if x == 0:
            c_pnl_AC += float(book_size * (train_predict[x]) * trainY_raw[x] - book_size * abs(
                train_predict[x]) * tc_per_trade_per_coin)
        else:
            #Susequent trading requires the deduction of TC for the change in the position
            c_pnl_AC += float(book_size * (train_predict[x]) * trainY_raw[x] - book_size * abs(
                train_predict[x] - train_predict[x - 1]) * tc_per_trade_per_coin)

        is_pnl_series_AC.append(c_pnl_AC)
        valid_pnl_series_AC.append(c_pnl_AC)
        os_pnl_series_AC.append(c_pnl_AC)
        
    # After cost analysis on CV data
    for x in range(validY.shape[0]):
        if x == 0:
            c_pnl_AC += float(book_size * (valid_predict[x]) * validY_raw[x] - book_size * abs(
                valid_predict[x] - train_predict[-1]) * tc_per_trade_per_coin)
        else:
            c_pnl_AC += float(book_size * (valid_predict[x]) * validY_raw[x] - book_size * abs(
                valid_predict[x] - valid_predict[x - 1]) * tc_per_trade_per_coin)

        valid_pnl_series_AC.append(c_pnl_AC)
        os_pnl_series_AC.append(c_pnl_AC)

    # After cost analysis on Test data
    for x in range(testY.shape[0]):
        if x == 0:
            c_pnl_AC += float(book_size * (test_predict[x]) * testY_raw[x] - book_size * abs(
                test_predict[x] - valid_predict[-1]) * tc_per_trade_per_coin)
        else:
            c_pnl_AC += float(book_size * (test_predict[x]) * testY_raw[x] - book_size * abs(
                test_predict[x] - test_predict[x - 1]) * tc_per_trade_per_coin)

        os_pnl_series_AC.append(c_pnl_AC)

    dates = [pd.to_datetime(x).date() for x in dates]
    # dates = np.arange(len(os_bc))
    step = 1000
    
    #Sending the data over to plotting function 
    TC_plotter(is_pnl_series_BC, valid_pnl_series_BC, os_pnl_series_BC, is_pnl_series_AC, 
               valid_pnl_series_AC, os_pnl_series_AC, dates, step, switch)

    return (is_pnl_series_BC, valid_pnl_series_BC, os_pnl_series_BC, is_pnl_series_AC, \
           valid_pnl_series_AC, os_pnl_series_AC)
