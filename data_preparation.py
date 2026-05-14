import os
import pandas as pd
from datasetsforecast.m5 import M5
from datasetsforecast.long_horizon import LongHorizon

def split_train_test(df, horizon):
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    test = df.groupby('unique_id').tail(horizon)
    train = df.drop(test.index)
    return train, test

def main():
    os.makedirs('data', exist_ok=True)
    
    print("Loading M5 Dataset...")
    Y_df_m5, *_ = M5.load('./data')
    Y_df_m5 = Y_df_m5[Y_df_m5['unique_id'].astype(str).str.contains('HOBBIES_1')]
    train_m5, test_m5 = split_train_test(Y_df_m5, 24)
    train_m5.to_csv('data/m5_hobbies_train.csv', index=False)
    test_m5.to_csv('data/m5_hobbies_test.csv', index=False)
    print("M5 done.")
    
    print("Loading ETTh1 Dataset...")
    Y_df_ett, *_ = LongHorizon.load('./data', 'ETTh1')
    train_ett, test_ett = split_train_test(Y_df_ett, 96)
    train_ett.to_csv('data/etth1_train.csv', index=False)
    test_ett.to_csv('data/etth1_test.csv', index=False)
    print("ETTh1 done.")
    
    print("Loading AirPassengers Dataset...")
    url = "https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv"
    try:
        Y_df_air = pd.read_csv(url)
    except Exception:
        # Fallback to standard URL
        url = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
        Y_df_air = pd.read_csv(url)
        
    if 'unique_id' not in Y_df_air.columns:
        if 'Month' in Y_df_air.columns and '#Passengers' in Y_df_air.columns:
            Y_df_air.rename(columns={'Month': 'ds', '#Passengers': 'y'}, inplace=True)
        Y_df_air.insert(0, 'unique_id', 'AirPassengers')
        
    Y_df_air = Y_df_air[['unique_id', 'ds', 'y']]
    
    train_air, test_air = split_train_test(Y_df_air, 12)
    train_air.to_csv('data/airpassengers_train.csv', index=False)
    test_air.to_csv('data/airpassengers_test.csv', index=False)
    print("AirPassengers done.")

if __name__ == '__main__':
    main()
