import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

def calculate_all_metrics(y_true, y_pred):
    # Sıfıra bölünme hatalarını engellemek için küçük bir epsilon
    epsilon = 1e-10
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # MAPE: Gerçek değer sıfırsa anlamsızlaşır, bu yüzden limit koyuyoruz
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_val = np.abs((y_true - y_pred) / (y_true + epsilon))
        mape = np.mean(mape_val[np.isfinite(mape_val)]) * 100
        # Eğer MAPE absürt bir sayıysa (M5 gibi) 'N/A' veya çok büyük bir sınır koy
        mape = mape if mape < 1e6 else np.inf

    smape = 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))
    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + epsilon) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE %': round(mape, 2),
        'sMAPE %': round(smape, 2),
        'WAPE %': round(wape, 2),
        'R2 Score': round(r2, 4)
    }

def generate_final_benchmark():
    os.makedirs('reports', exist_ok=True)
    datasets = ['m5_hobbies', 'etth1', 'airpassengers']
    
    # Tüm potansiyel modeller (model_training.py ve chronos_benchmark.py çıktıları)
    models = ['AutoARIMA', 'AutoETS', 'NBEATS', 'LSTM', 'GRU', 'Chronos']
    summary_list = []

    for name in datasets:
        print(f"📊 Rapor Hazırlanıyor: {name.upper()}")
        
        try:
            # Dosyaları yükle (model_training.py artık _preds.csv olarak kaydediyor)
            base_path = f'predictions/{name}_preds.csv'
            chronos_path = f'predictions/{name}_chronos_only.csv'
            
            if not os.path.exists(base_path) or not os.path.exists(chronos_path):
                print(f"⚠️ {name} için dosyalar eksik, atlanıyor...")
                continue

            base_df = pd.read_csv(base_path)
            chronos_df = pd.read_csv(chronos_path)
            
            # Format eşitleme
            for df in [base_df, chronos_df]:
                df['unique_id'] = df['unique_id'].astype(str)
                df['ds'] = pd.to_datetime(df['ds'])

            # Merge
            full_df = pd.merge(base_df, chronos_df, on=['unique_id', 'ds'], how='inner')
            
            # 1. Metrik Hesaplama
            for model in models:
                if model in full_df.columns:
                    metrics = calculate_all_metrics(full_df['y'], full_df[model])
                    metrics['Dataset'] = name
                    metrics['Model'] = model
                    summary_list.append(metrics)

            # 2. Görselleştirme (Daha İyi Grafik)
            plt.figure(figsize=(14, 7))
            # Her veri setinden rastgele/ilk başarılı örneği al
            sample_id = full_df['unique_id'].unique()[0]
            sample_df = full_df[full_df['unique_id'] == sample_id].sort_values('ds')

            plt.plot(sample_df['ds'], sample_df['y'], label='Gerçek (Actual)', color='black', linewidth=2.5, zorder=3)
            
            for model in models:
                if model in sample_df.columns:
                    plt.plot(sample_df['ds'], sample_df[model], label=model, linestyle='--', alpha=0.8)

            plt.title(f"Benchmark: {name.upper()} | Örnek ID: {sample_id}")
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'reports/{name}_comparison.png')
            plt.close()

        except Exception as e:
            print(f"❌ {name} işlenirken hata: {e}")

    # Final Tablo Kaydı
    if summary_list:
        report_df = pd.DataFrame(summary_list)
        report_df = report_df[['Dataset', 'Model', 'MAE', 'RMSE', 'MAPE %', 'sMAPE %', 'WAPE %', 'R2 Score']]
        report_df.to_csv('reports/complete_benchmark_results.csv', index=False)
        
        print("\n" + "="*50)
        print("🏆 NİHAİ SONUÇLAR")
        print("="*50)
        print(report_df.to_string(index=False))
    else:
        print("🚨 Hiç sonuç üretilemedi! Lütfen tahmin dosyalarını kontrol et.")

if __name__ == '__main__':
    generate_final_benchmark()