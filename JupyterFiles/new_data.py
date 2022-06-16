import dask.dataframe as dd
import pandas as pd

binance = dd.read_csv('/home/donlekout/Desktop/Data/kaiko-trades/gz_v1/Binance/BTCUSDT/**/*.csv.gz',\
 engine = 'c', compression = 'gzip', assume_missing=True, blocksize=None)
binance = binance.compute(num_workers = 20)

binance.drop(['id', 'exchange', 'symbol'], inplace = True, axis = 'columns')

binance.sort_values(by = 'date', inplace = True)

dd_binance=dd.from_pandas(binance, npartitions = 200)

dd_binance.to_parquet('/home/donlekout/Desktop/Thesis/Data/Binance/BTCUSDT/', engine = 'fastparquet')
