from tabicl import TabICLClassifier

import numpy as np
import pandas as pd
import time
import torch
import resource
import psutil
import os

start = time.time()

# Reprodutibilidade
np.random.seed(42)

n_samples = 6400
n_features = 100
n_classes = 2
# X_train: dados aleatórios contínuos
X_train = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f"feature_{i}" for i in range(n_features)]
)
# y_train: classes inteiras de 0 a 9
y_train = np.random.randint(0, n_classes, size=n_samples)

X_test = pd.DataFrame(
    np.random.randn(2000, n_features),
    columns=[f"feature_{i}" for i in range(n_features)]
)

ini = time.time()
clf = TabICLClassifier(batch_size=4)
clf.fit(X_train, y_train)  # downloads checkpoint on first use, otherwise cheap
pred = clf.predict(X_test)  # in-context learning happens here


end = time.time()
elapsed = end - start
peak_gpu = torch.cuda.max_memory_allocated()/1024**3
peak_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

print("\n========= FINAL STATS =========")
print(f"Time: {elapsed/60:.2f} min")
print(f"Peak GPU RAM: {peak_gpu:.2f} GB")
print(f"Peak CPU RAM: {peak_cpu:.2f} MB")
print("===============================")