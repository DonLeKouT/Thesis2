import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt

def dataset_creation_2021(data):
    '''
    Resampling to 1 second and aggregating trades, sells, volume
    '''
    from_date = '2021-01-01 00:00:00.000'
    dataset = data.loc[from_date:, :].copy()

    dataset.interpolate(method = 'spline', order = 3, inplace = True)

    #Weighted Average
    aggr_amount = dataset['amount'].resample('S').sum().copy()
    temp = (dataset['amount'] * dataset['price']).resample('S').sum()
    weighted_average = temp / aggr_amount

    #Trades
    dataset['ones'] = 1
    trades = dataset['ones'].resample('S').sum()

    #Sell
    sells = dataset['sell'].resample('S').sum()

    #Percentage of Sells
    percent_sell = sells / trades

    final = pd.concat((weighted_average, aggr_amount, trades, sells, percent_sell), axis = 1)
    final.columns = ['wp', 'amount', 'trades', 'sell', 'sell_perc']

    return final



def dataset_creation_2020(data):
    '''
    Resampling to 1 second and aggregating trades, sells, volume
    '''
    data.interpolate(method = 'spline', order = 3, inplace = True)

    #Weighted Average
    aggr_amount = data['amount'].resample('S').sum().copy()
    temp = (data['amount'] * data['price']).resample('S').sum()
    weighted_average = temp / aggr_amount

    #Trades
    data['ones'] = 1
    trades = data['ones'].resample('S').sum()

    #Sell
    sells = data['sell'].resample('S').sum()

    #Percentage of Sells
    percent_sell = sells / trades

    final = pd.concat((weighted_average, aggr_amount, trades, sells, percent_sell), axis = 1)
    final.columns = ['wp', 'amount', 'trades', 'sell', 'sell_perc']

    return final


    

def create_aggregate():
    '''
    Create an aggregated dataset from 2018 to 2021. Resampled to the second. 
    '''
    data2018_2019 = pd.read_parquet('/home/donlekout/Desktop/Thesis/Data/AGGR/spot_dataset_second_2018_2019_2.parquet')

    tara = data2018_2019.copy()
    taraindex = data2018_2019.columns
    tara.columns = taraindex.swaplevel()
    #tara.dropna(inplace = True)

    aggregated2018_2019 = pd.DataFrame(tara['trades'].sum(axis = 1), columns = ['trades'])
    aggregated2018_2019['amount'] = tara['amount'].sum(axis=1)
    aggregated2018_2019['sell'] = tara['sell'].sum(axis = 1)
    aggregated2018_2019['sell_volume'] = tara['sell_volume'].sum(axis = 1)
    aggregated2018_2019['price'] = (tara['amount'].div(tara['amount'].sum(axis=1), axis=0)).multiply(tara['wp'], axis=0).sum(axis = 1)
    aggregated2018_2019['premium'] = tara['wp'][['coinbase_btcusd', 'bitstamp', 'kraken_btcusd']].multiply(tara['amount'][['coinbase_btcusd', 'bitstamp', 'kraken_btcusd']].divide(tara['amount'][['coinbase_btcusd', 'bitstamp', 'kraken_btcusd']].sum(axis = 1), axis = 0), axis = 0).sum(axis = 1) - tara['wp'][['binance', 'huobi_btcusdt']].multiply(tara['amount'][['binance', 'huobi_btcusdt']].divide(tara['amount'][['binance', 'huobi_btcusdt']].sum(axis = 1), axis = 0), axis = 0).sum(axis = 1)
    del tara

    data2020 = pd.read_parquet('/home/donlekout/Desktop/Thesis/Data/AGGR/spot_dataset_second_2020_2.parquet')

    tara = data2020.copy()
    taraindex = data2020.columns
    tara.columns = taraindex.swaplevel()
    #tara.dropna(inplace = True)

    aggregated2020 = pd.DataFrame(tara['trades'].sum(axis = 1), columns = ['trades'])
    aggregated2020['amount'] = tara['amount'].sum(axis=1)
    aggregated2020['sell'] = tara['sell'].sum(axis = 1)
    aggregated2020['sell_volume'] = tara['sell_volume'].sum(axis = 1)
    aggregated2020['price'] = (tara['amount'].div(tara['amount'].sum(axis=1), axis=0)).multiply(tara['wp'], axis=0).sum(axis = 1)
    aggregated2020['premium'] = tara['wp'][['coinbase_btcusd', 'bitstamp', 'kraken_btcusd']].multiply(tara['amount'][['coinbase_btcusd', 'bitstamp', 'kraken_btcusd']].divide(tara['amount'][['coinbase_btcusd', 'bitstamp', 'kraken_btcusd']].sum(axis = 1), axis = 0), axis = 0).sum(axis = 1) - tara['wp'][['binance', 'huobi_btcusdt']].multiply(tara['amount'][['binance', 'huobi_btcusdt']].divide(tara['amount'][['binance', 'huobi_btcusdt']].sum(axis = 1), axis = 0), axis = 0).sum(axis = 1)
    del tara

    data2021 = pd.read_parquet('/home/donlekout/Desktop/Thesis/Data/AGGR/spot_dataset_second_2021.parquet')

    tara = data2021.copy()
    taraindex = data2021.columns
    tara.columns = taraindex.swaplevel()
    #tara.dropna(inplace = True)

    aggregated2021 = pd.DataFrame(tara['trades'].sum(axis = 1), columns = ['trades'])
    aggregated2021['amount'] = tara['amount'].sum(axis=1)
    aggregated2021['sell'] = tara['sell'].sum(axis = 1)
    aggregated2021['sell_volume'] = tara['sell_volume'].sum(axis = 1)
    aggregated2021['price'] = (tara['amount'].div(tara['amount'].sum(axis=1), axis=0)).multiply(tara['wp'], axis=0).sum(axis = 1)
    aggregated2021['premium'] = tara['wp'][['coinbase_btcusd', 'bitstamp', 'kraken_btcusd']].multiply(tara['amount'][['coinbase_btcusd', 'bitstamp', 'kraken_btcusd']].divide(tara['amount'][['coinbase_btcusd', 'bitstamp', 'kraken_btcusd']].sum(axis = 1), axis = 0), axis = 0).sum(axis = 1) - tara['wp'][['binance', 'huobi_btcusdt']].multiply(tara['amount'][['binance', 'huobi_btcusdt']].divide(tara['amount'][['binance', 'huobi_btcusdt']].sum(axis = 1), axis = 0), axis = 0).sum(axis = 1)
    del tara

    del data2018_2019, data2020, data2021

    aggregated = pd.concat((aggregated2018_2019, aggregated2020, aggregated2021), axis = 0)
    aggregated['buy_volume'] = aggregated['amount'] - aggregated['sell_volume']
    aggregated['buy'] = aggregated['trades'] - aggregated['sell']
    aggregated['buy-sell_volume_perc'] = (aggregated['buy_volume'] - aggregated['sell_volume']).divide(aggregated['amount'], axis=0) 
    aggregated['buy-sell_trades_perc'] = (aggregated['buy'] - aggregated['sell']).divide(aggregated['trades'], axis=0) 
    del aggregated2018_2019, aggregated2021, aggregated2020

    return aggregated



def bull_bear_flat(aggregated):
    '''
    Create 3 dataframes with multiindex: One for bull periods, one for bear and one for flat. 
    '''
    
    bull1 = aggregated.loc['2020-12-15 00:00:30' : '2021-01-08 00:00:30', :].copy()
    bull2 = aggregated.loc['2019-02-08 00:00:30' : '2019-06-26 00:00:30', :].copy()
    bull3 = aggregated.loc['2020-03-14 00:00:30' : '2020-09-02 00:00:30', :].copy()
    bull = pd.concat((bull1, bull2, bull3), axis = 0, keys = ['bull1', 'bull2', 'bull3'])
    del bull1, bull2, bull3


    bear1 = aggregated.loc['2018-02-18 00:00:30' : '2018-12-15 00:00:30', :].copy()
    bear2 = aggregated.loc['2020-02-13 00:00:30' : '2020-03-13 00:00:30', :].copy()
    bear3 = aggregated.loc['2019-06-27 00:00:30' : '2020-01-02 00:00:30', :].copy()
    bear4 = aggregated.loc['2021-04-14 00:00:30' : '2021-05-23 00:00:30', :].copy()
    bear = pd.concat((bear1, bear2, bear3, bear4), axis = 0, keys = ['bear1', 'bear2', 'bear3', 'bear4'])
    del bear1, bear2, bear3, bear4


    flat1 = aggregated.loc['2018-04-01 00:00:30' : '2018-12-21 00:00:30', :].copy()
    flat2 = aggregated.loc['2020-04-30 00:00:30' : '2020-07-20 00:00:30', :].copy()
    flat3 = aggregated.loc['2020-07-28 00:00:30' : '2020-09-02 00:00:30', :].copy()
    flat4 = aggregated.loc['2021-02-10 00:00:30' : '2021-05-13 00:00:30', :].copy()
    flat5 = aggregated.loc['2021-05-22 00:00:30' : '2021-07-20 00:00:30', :].copy()
    flat = pd.concat((flat1, flat2, flat3, flat4, flat5), axis = 0, keys = ['flat1', 'flat2', 'flat3', 'flat4', 'flat5'])
    del flat1, flat2, flat3, flat4, flat5

    return bull, bear, flat


def no_duplicates(data):
    '''
    Aggregates data with respect to the timestamp. Uses groupby with index.
    '''
    data['ones'] = 1

    price = data['price'].groupby(level=0).last()
    
    agg = data[['sell', 'ones', 'amount']].groupby(level=0).sum()
    #sells = data['sell'].groupby(level=0).sum()
    #trades = data['ones'].groupby(level = 0).sum()
    #amount = data['amount'].groupby(level = 0).sum()
    
    sell_amount = (data['sell'] * data['amount']).groupby(level=0).sum()
    dollars = (data['price'] * data['amount']).groupby(level=0).sum()

    data = pd.concat((price, agg, sell_amount, dollars), axis = 1)
    data.columns = ['price', 'sells', 'trades', 'amount', 'sell_amount', 'dollars']
    return(data)


def generate_volumebars(trades, frequency=10):
    times = trades[:,0]
    prices = trades[:,1]
    volumes = trades[:,2]
    ans = np.zeros(shape=(len(prices), 6))
    candle_counter = 0
    vol = 0
    lasti = 0
    for i in range(len(prices)):
        vol += volumes[i]
        if vol >= frequency:
            ans[candle_counter][0] = times[i]                          # time
            ans[candle_counter][1] = prices[lasti]                     # open
            ans[candle_counter][2] = np.max(prices[lasti:i+1])         # high
            ans[candle_counter][3] = np.min(prices[lasti:i+1])         # low
            ans[candle_counter][4] = prices[i]                         # close
            ans[candle_counter][5] = np.sum(volumes[lasti:i+1])        # volume
            candle_counter += 1
            lasti = i+1
            vol = 0
    return ans[:candle_counter]


def generate_volumebars2(trades, frequency=10):
    
    ans = np.zeros(shape=(len(trades)))
    candle_counter = 1
    vol = 0
    
    for i in range(len(trades)):
        vol += trades[i]
        ans[i] = candle_counter
        if vol >= frequency:
            candle_counter += 1
            vol = 0

    return ans



@jit(nopython=True)    
def tib(b_t, initial_imbalance, alpha):

    weighted_count = 0 # Denominator for normalization of EWMAs
    weighted_sum_T = 0 # Nominator for EWMA of duration of a bar
    weighted_sum_imbalance = 0 # Nominator for EWMA of the imbalance of bar

    out = np.zeros(b_t.shape)
    dummy = 0
    imbalance = initial_imbalance
    T = 0
    debug = np.zeros(b_t.shape)

    for i in range(len(b_t)):
        dummy += b_t[i]
        debug[i] = dummy
        T += 1
        if (abs(dummy)>=imbalance):
            out[i] = 1
            weighted_sum_T = T + (1-alpha)*weighted_sum_T
            weighted_sum_imbalance = dummy/(1.0*T) + (1 - alpha) * weighted_sum_imbalance
            weighted_count = 1 + (1-alpha) * weighted_count
            ewma_T = weighted_sum_T/weighted_count
            ewma_imbalance = weighted_sum_imbalance/weighted_count
            imbalance = ewma_T * abs(ewma_imbalance)
            dummy = 0
            T = 0

    return out



@jit(nopython=True)    
def tib2(b_t, initial_imbalance, alpha):

    weighted_sum_T = 0 # EWMA of duration of a bar
    weighted_sum_Probs = 0 # EWMA of Probabilities

    out = np.zeros(b_t.shape)
    dummy = 0
    imbalance = initial_imbalance
    T = 0

    for i in range(len(b_t)):
        dummy += b_t[i]
        T += 1
        if (abs(dummy)>=imbalance):
            out[i] = 1
            weighted_sum_T = alpha*T + (1-alpha)*weighted_sum_T
            weighted_sum_Probs = alpha*np.sum(b_t[i-T:i] == 1)/T + (1-alpha)*weighted_sum_Probs
            imbalance = weighted_sum_T * abs(2*weighted_sum_Probs - 1)
            dummy = 0
            T = 0

    return out




def TIB(df,column,initial_imbalance,alpha):

    weighted_sum_T = 0 
    weighted_sum_prob = 0 
    df["delta_p"] = df[column].pct_change()
    imbalance = initial_imbalance
    b_t = np.zeros(len(df["delta_p"]))
    b_t[0]=0
    indx = []
    T=0

    for i in range(1,len(df["delta_p"])):
        if df["delta_p"][i] == 0: 
            b_t[i]=b_t[i-1]
        else:
            b_t[i] = np.abs(df["delta_p"][i])/df["delta_p"][i]
        T+=1
        if (abs(b_t.sum())>=imbalance):
            indx.append(i)
            weighted_sum_T = alpha*T + (1-alpha)*weighted_sum_T
            weighted_sum_prob = alpha*sum(x>0 for x in b_t)/(T) + (1 - alpha) * weighted_sum_prob
            imbalance = weighted_sum_T * abs(2*weighted_sum_prob-1)
        
            T = 0
            b_t = np.zeros(len(df["delta_p"]))
    
    return indx


jit(nopython=True, nogil = True)    
def tib3(b_t, initial_imbalance, alpha):

    weighted_sum_T = 0 # EWMA of duration of a bar
    weighted_sum_Probs = 0 # EWMA of Probabilities

    index = np.zeros(len(b_t))
    b_t_sum = 0
    imbalance = initial_imbalance
    T = 0
    count_probs = 0

    for i in range(len(b_t)):
        b_t_sum += b_t[i]
        T += 1
        count_probs += (b_t[i] == 1)
        if (np.abs(b_t_sum)>=imbalance):
            index[i] = 1
            weighted_sum_T = alpha*T + (1-alpha)*weighted_sum_T
            weighted_sum_Probs = alpha*count_probs/T + (1-alpha)*weighted_sum_Probs
            imbalance = weighted_sum_T * np.abs(2*weighted_sum_Probs - 1)
            b_t_sum = 0
            T = 0
            count_probs = 0

    return index



def plot_sample(dataset, initial, alpha):
    kappa = imbalance_volume_bars(dataset['b_t'].to_numpy(), initial, alpha)
    np.sum(kappa)

    lol = pd.concat((dataset['price'].reset_index(), pd.Series(kappa)), ignore_index=True, axis = 1)
    lol.columns = ['date', 'price', 'kappa']

    plt.plot(dataset['price'].resample('H').mean())
    plt.scatter(lol.loc[lol['kappa'] == 1]['date'], lol.loc[lol['kappa'] == 1]['price'], label = 'kappa')
    plt.xticks(rotation = 45)
    plt.legend()
    plt.show()



@jit(nopython=True, nogil = True)   
def imbalance_volume_bars(volume, initial_imbalance, alpha):

    index = np.zeros(len(volume))
    index_plus = []
    index_minus = []
    temp_plus = 0
    temp_minus = 0
    imbalance = initial_imbalance
    ewma_plus = 0
    ewma_minus = 0
    debug = []

    for i in range(len(volume)):
        
        temp_plus += volume[i] * (volume[i] > 0)
        temp_minus += volume[i] * (volume[i] < 0)
    
        if np.abs(temp_plus + temp_minus) > np.abs(imbalance):
            index[i] = np.sign(temp_plus + temp_minus)
            index_plus.append(temp_plus)
            index_minus.append(temp_minus)
            debug.append(imbalance)

            ewma_plus = alpha * temp_plus + (1 - alpha) * ewma_plus
            ewma_minus = alpha * temp_minus + (1 - alpha) * ewma_minus

            #imbalance = ewma_plus - ewma_minus
            #sum1 = 0
            #sum2 = 0
            #for i in range(7):
                #sum1 += index_plus[-7:][i]
                #sum2 += index_minus[-7:][i]
            
            imbalance = ewma_plus + ewma_minus

            temp_plus = 0
            temp_minus = 0
            
    return index, index_plus, index_minus, debug



@jit(nopython = True, nogil = True)
def maximum(dataset, k):
    n = len(dataset)
    out = np.zeros(n)
    index = 0

    for i in range(n-k):
        if ((dataset[i] - dataset[k]) > -0.2) & ((dataset[i] - dataset[k]) < 0.2):
            index = int((2*i + k)/2)
            out[index] = 1

    return out



@jit(nogil = True)
def tiel4(dataset, length):
    n = dataset.shape[0]
    out = np.zeros((dataset.shape[0]))
    test = np.zeros(length)
    temp = np.zeros(length)

    for i in np.arange(length,n,5):
        temp = dataset[i-length : i]
        argmax_ = np.argmax(temp)
        argmin_ = np.argmin(temp)
        last = np.maximum(argmax_, argmin_)
        first = np.minimum(argmax_, argmin_)
        if (last - first)/len(temp) > 0.15:
            test = temp[first : last]
        else:
            pass


        S_t = 0
        for j in np.arange(len(test)):
            returns = np.log(np.diff(test, n=1))
            S_t += np.minimum(0, S_t - returns[j] + returns[j-1])
        
        out[i] = S_t if S_t < -0.1 else 0
    return out


@jit(nogil = True)
def tiel3(dataset, length):
    n = dataset.shape[0]
    out = np.zeros((dataset.shape[0]))
    test = np.zeros(length)
    temp = np.zeros(length)

    for i in np.arange(length,n,5):
        temp = dataset[i-length : i]
        argmax_ = np.argmax(temp)
        argmin_ = np.argmin(temp)
        last = np.maximum(argmax_, argmin_)
        first = np.minimum(argmax_, argmin_)
        if (last - first)/len(temp) > 0.15:
            test = temp[first : last]
        else:
            pass


        S_t = 0
        for j in np.arange(len(test)):
            returns = np.log(np.diff(test, n=1))
            S_t += np.maximum(0, S_t + returns[j] - returns[j-1])
        
        out[i] = S_t if S_t > 0.1 else 0
    return out


@jit(nogil = True)
def tiel2(dataset, length):
    n = dataset.shape[0]
    out = np.zeros((dataset.shape[0],2))
    test = np.zeros(length)
    temp = np.zeros(length)

    for i in np.arange(length,n,5):
        temp = dataset[i-length : i]
        argmax_ = np.argmax(temp)
        argmin_ = np.argmin(temp)
        last = np.maximum(argmax_, argmin_)
        first = np.minimum(argmax_, argmin_)
        if (last - first)/len(temp) > 0.15:
            test = temp[first : last]
        else:
            pass


        S_t = 0
        S_t_ = 0
        for j in np.arange(len(test)):
            returns = np.log(np.diff(test, n=1))
            S_t += np.maximum(0, S_t + returns[j] - returns[j-1])
            S_t_ += np.minimum(0, - S_t_ - returns[j] + returns[j-1])
        
        out[i,0] = S_t > 0.1
        out[i,1] = S_t_ < -0.1
    return out


@jit(nopython = True, nogil = True)
def imbalance_ticks(dataset, alpha, beta, gamma, initial): #Last Attempt
    n = dataset.shape[0]
    out = np.zeros(n)
    ewma1 = 5
    ewma2 = 5
    theta = 0
    t_plus = 0
    T = 0
    imbalance = initial
    weighted_count_1 = 0
    weighted_count_2 = 0
    debug = np.zeros((n,3))

    for i in np.arange(n):
        T +=1
        theta += dataset[i]
        t_plus += dataset[i] == 1
        if np.abs(theta) >= imbalance:
            weighted_count_1 = (1 - (1 - alpha)**T) / alpha
            weighted_count_2 = (1 - (1 - beta)**T) / beta
            weighted_count_3 = (1 - (1 - gamma)**T) / gamma
            ewma1 = (T + (1-alpha) * ewma1) / weighted_count_1
            ewma2 = (np.abs(((2 * t_plus / T) - 1)) + (1-beta) * ewma2) / weighted_count_2
            imbalance = (np.abs(ewma1 * ewma2) + (1-gamma) * imbalance) / weighted_count_3 
            T = 0
            t_plus = 0
            out[i] = np.sign(theta)
            theta = 0
            debug[i,] = [imbalance, ewma1, ewma2]
        
    return out, debug


@jit(nopython = True, nogil = True)
def helpfull(dataset, from_, to):
    index = dataset
    results = []
    n = dataset.shape[0]
    for i in np.arange(n):
        if index[i] > from_:
            results.append(i)
            start_int = i
            break
    
    for j in np.arange(start_int,n):
        if index[j] > to:
            results.append(j)
            break

    return results

@jit()
def slicing(dataset, from_, to):
    index = pd.to_numeric(dataset.index).values
    start = pd.to_datetime([from_]).astype(int)[0]
    end = pd.to_datetime([to]).astype(int)[0]

    out = helpfull(index, start, end)

    return out



@jit(nopython = True, nogil = True)
def imbalance(dataset, slow, fast, state, threshold):
    n = dataset.shape[0]
    out = np.zeros(n)
    moving_slow = np.sum(dataset[:slow])
    moving_fast = np.sum(dataset[slow - fast:slow])
    state = state
    temp = 1
    #debug = np.zeros((n,2))

    for i in np.arange(slow+1,n):
        if temp != state:
            out[i] = 1
            temp = state
        
        moving_slow = (moving_slow + dataset[i] - dataset[i-slow])/slow
        moving_fast = (moving_fast + dataset[i] - dataset[i-fast])/fast
        if moving_fast - moving_slow >= threshold:
            state = 1
        elif moving_fast - moving_slow < -threshold:
            state = 0
        else:
            state = 3
        #debug[i,] = [moving_slow, moving_fast]

    return out


@jit(nopython = True, nogil = True)
def imbalance_exp(dataset, alpha, beta, state, threshold):
    n = dataset.shape[0]
    out = np.zeros(n)
    ewma_slow = 0
    ewma_fast = 0
    state = state
    temp = 1
    #debug = np.zeros((n,2))

    for i in np.arange(n):
        if temp != state:
            out[i] = 1
            temp = state
        
        ewma_slow = alpha * dataset[i] + (alpha - 1) * ewma_slow
        ewmq_fast = beta * dataset[i] + (beta - 1) * ewma_fast
        if ewmq_fast - ewma_slow >= threshold:
            state = 1
        elif ewmq_fast - ewma_slow < -threshold:
            state = 0
        else:
            state = 3
        #debug[i,] = [ewmq_fast, ewma_slow]

    return out


@jit(nopython = True, nogil = True)
def imbalance_ticks2(dataset, alpha, beta, gamma, initial):
    n = dataset.shape[0]
    out = np.zeros(n)
    ewma1 = 5
    ewma2 = 5
    theta = 0
    t_plus = 0
    T = 0
    imbalance = initial
    weighted_count_1 = 0
    weighted_count_2 = 0
    #debug = np.zeros((n,3))

    for i in np.arange(n):
        T +=1
        theta += dataset[i]
        t_plus += dataset[i] == 1
        if (np.abs(theta) >= imbalance) & (T > 500):
            weighted_count_1 = (1 - (1 - alpha)**T) / alpha
            weighted_count_2 = (1 - (1 - beta)**T) / beta
            weighted_count_3 = (1 - (1 - gamma)**T) / gamma
            ewma1 = (T + (1-alpha) * ewma1) / weighted_count_1
            ewma2 = (np.abs(((2 * t_plus / T) - 1)) + (1-beta) * ewma2) / weighted_count_2
            imbalance = (np.abs(ewma1 * ewma2) + (1-gamma) * imbalance) / weighted_count_3 
            T = 0
            t_plus = 0
            out[i] = np.sign(theta)
            theta = 0
            #debug[i,] = [imbalance, ewma1, ewma2]
        
    return out


@jit(nopython = True, nogil = True)
def imbalance_filters(dataset, high, threshold, distance):
    n = dataset.shape[0]
    out = np.zeros(n)
    low_pass = np.ones(high) / high
    T = 0

    for i in np.arange(high+1, n, 1):
        T += 1
        new = np.array([dataset[i-high-1:i-1] @ low_pass, dataset[i-high:i] @ low_pass])
        difference = np.diff(new[-2:])

        if ((np.abs(difference)[0]) >= threshold) & (T > distance):
            out[i] = 1
            T = 0
        
    return out


@jit(nopython = True, nogil = True)
def imbalance_exp2(dataset, alpha, beta, threshold, summary = False):
    n = dataset.shape[0]
    out = np.zeros(n)
    ewma_slow = 0
    ewma_fast = 0
    state = 1
    temp = 1
    summary = 0
    count = 1

    for i in np.arange(n):
        if temp != state:
            out[i] = count * (~(count % 2 == 0))
            count += 1
            temp = state
            summary = 0
        
        if summary:
            summary += dataset[i]
        else:
            summary = dataset[i]

        ewma_slow = alpha * summary + (alpha - 1) * ewma_slow
        ewma_fast = beta * summary + (beta - 1) * ewma_fast
        if ewma_fast - ewma_slow >= threshold:
            state = 1
        elif ewma_fast - ewma_slow < -threshold:
            state = 0
        else:
            state = 3

    return out