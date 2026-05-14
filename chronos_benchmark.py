import os
import time
import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline

def run_adaptive_chronos():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        dtype=torch.float16, 
    )

    datasets = [
        {'name': 'm5_hobbies', 'h': 24, 'freq': 'D'},
        {'name': 'etth1', 'h': 96, 'freq': 'H'},
        {'name': 'airpassengers', 'h': 12, 'freq': 'MS'}
    ]

    # Başlangıç batch boyutu
    INITIAL_BATCH_SIZE = 64 

    for ds in datasets:
        name, h, freq = ds['name'], ds['h'], ds['freq']
        df_path = f'data/{name}_train.csv'
        if not os.path.exists(df_path): continue
        
        print(f"\nVERİ SETİ: {name.upper()}")
        train_df = pd.read_csv(df_path)
        start_time = time.time()

        grouped = train_df.groupby('unique_id')['y'].apply(list).to_dict()
        uids = list(grouped.keys())
        all_series = [torch.tensor(v) for v in grouped.values()]
        
        all_forecasts = []
        batch_size = INITIAL_BATCH_SIZE
        i = 0

        while i < len(all_series):
            try:
                batch = all_series[i : i + batch_size]
                with torch.no_grad():
                    forecasts = pipeline.predict(batch, h)
                    all_forecasts.extend(forecasts.numpy())
                i += batch_size
                if i % 128 == 0 or i >= len(all_series):
                    print(f"İlerleme: {min(i, len(uids))} / {len(uids)} (Batch Size: {batch_size})")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️  OOM Hatası! Batch boyutu {batch_size}'den {batch_size // 2}'ye düşürülüyor...")
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    if batch_size < 1:
                        print("🚨 Kritik Hata: Batch boyutu 1 bile sığmıyor!")
                        break
                else:
                    raise e 

        # Sonuçları Kaydet
        last_dates = train_df.groupby('unique_id')['ds'].max().map(pd.to_datetime).to_dict()
        results = []
        for j, uid in enumerate(uids):
            median_forecast = np.quantile(all_forecasts[j], 0.5, axis=0)
            dates = pd.date_range(start=last_dates[uid], periods=h+1, freq=freq)[1:]
            results.append(pd.DataFrame({'unique_id': uid, 'ds': dates, 'Chronos': median_forecast}))

        duration = time.time() - start_time
        pd.concat(results).to_csv(f'predictions/{name}_chronos_only.csv', index=False)
        print(f"{name} Bitti! Süre: {duration:.2f} sn.")

if __name__ == '__main__':
    run_adaptive_chronos()