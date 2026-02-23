from tabicl import TabICLRegressor

import numpy as np
import pandas as pd
import time
import torch
import resource

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

def add_mcar_noise(data, missing_rate, seed):
    """
    Adds MCAR (Missing Completely At Random) noise to a dataset.

    Args:
        data (torch.Tensor): The dataset as a tensor of shape (n_samples, height, width).
        missing_rate (float): Proportion of values to set as missing (0 <= missing_rate <= 1).

    Returns:
        noisy_data (torch.Tensor): Dataset with MCAR noise applied.
    """
    noisy_data = data.copy()
    # mask = np.random.random(len(noisy_data)) < missing_rate
    # seed = 0
    torch.manual_seed(seed)
    mask = torch.FloatTensor(noisy_data.shape[0], noisy_data.shape[1]).uniform_() < missing_rate
    noisy_data[mask] = np.nan
    return noisy_data, mask

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

for dataset in ['concrete', 'energy', 'housing', 'power',  'yacht']:
    print(dataset)
    df_np = np.loadtxt(f'/data/uci/{dataset}/data/data.txt')
    classe = pd.DataFrame(df_np[:, -1:])
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for missing_rate in [0.3,0.5,0.7,0.9]:
        print(missing_rate)
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df_np[:, :-1])
        missing_rate=missing_rate
        df, mask_df = add_mcar_noise(df, missing_rate, 0)
        df = pd.DataFrame(df)
        X = df.values

        maes = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

            X_train, X_test = X[train_idx], X[val_idx]
            y_train, y_test = classe.iloc[train_idx], classe.iloc[val_idx]

            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            start = time.time()
            reg = TabICLRegressor()
            reg.fit(X_train, y_train)
            pred = reg.predict(X_test)
            end = time.time()
            elapsed = end - start
            resultado = mae(y_test, pred)
            maes.append(resultado)

            peak_gpu = torch.cuda.max_memory_allocated()/1024**3
            peak_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

            print("\n========= FINAL STATS =========")
            print(f"Time: {elapsed/60:.2f} min")
            print(f"Peak GPU RAM: {peak_gpu:.2f} GB")
            print(f"Peak CPU RAM: {peak_cpu:.2f} MB")
            print("===============================")

        print("MAE:", np.mean(maes), np.std(maes))
        