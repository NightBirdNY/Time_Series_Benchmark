import os
import time
import pandas as pd
import multiprocessing
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, LSTM, GRU

class BenchmarkSystem:
    def __init__(self):
        self.n_cores = multiprocessing.cpu_count()
        os.makedirs('predictions', exist_ok=True)
        self.performance_logs = []

    def standardize_data(self, df, name):
        df['unique_id'] = df['unique_id'].astype(str)
        df['ds'] = pd.to_datetime(df['ds'])
        if 'airpassengers' in name.lower():
            df['ds'] = df['ds'].dt.to_period('M').dt.to_timestamp()
        else:
            df['ds'] = df['ds'].dt.normalize()
        return df

    def run_benchmark(self):
        datasets = [
            {'name': 'm5_hobbies', 'h': 24, 'freq': 'D'},
            {'name': 'etth1', 'h': 96, 'freq': 'H'},
            {'name': 'airpassengers', 'h': 12, 'freq': 'MS'}
        ]

        for ds in datasets:
            name, h, freq = ds['name'], ds['h'], ds['freq']
            print(f"\n🚀 TRAINING: {name.upper()}")

            train_df = pd.read_csv(f'data/{name}_train.csv')
            test_df = pd.read_csv(f'data/{name}_test.csv')
            train_df = self.standardize_data(train_df, name)
            test_df = self.standardize_data(test_df, name)

            # 1. Stats Models
            sf = StatsForecast(models=[AutoARIMA(), AutoETS()], freq=freq, n_jobs=self.n_cores)
            sf_preds = sf.forecast(df=train_df, h=h).reset_index()

            # 2. Neural Models
            nf = NeuralForecast(
                models=[NBEATS(h=h, input_size=2*h, max_steps=200),
                        LSTM(h=h, input_size=2*h, max_steps=200),
                        GRU(h=h, input_size=2*h, max_steps=200)],
                freq=freq
            )
            nf.fit(df=train_df)
            nf_preds = nf.predict().reset_index()

            # Merge & Save (Chronos hariç)
            sf_preds = self.standardize_data(sf_preds, name)
            nf_preds = self.standardize_data(nf_preds, name)
            
            final_preds = pd.merge(sf_preds, nf_preds, on=['unique_id', 'ds'], how='outer')
            final_output = pd.merge(test_df[['unique_id', 'ds', 'y']], final_preds, on=['unique_id', 'ds'], how='inner')
            
            # final_report.py ile uyumlu isim
            final_output.to_csv(f'predictions/{name}_preds.csv', index=False)
            print(f"✅ {name} eğitildi ve kaydedildi.")

if __name__ == '__main__':
    BenchmarkSystem().run_benchmark()