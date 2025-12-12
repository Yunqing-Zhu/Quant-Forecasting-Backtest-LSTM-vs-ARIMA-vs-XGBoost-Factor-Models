
import numpy as np
import pandas as pd
import argparse
import os

def generate(n=10000, n_features=20, anomaly_frac=0.02, random_state=42, out_path="data/transactions.csv"):
    rng = np.random.RandomState(random_state)
    # normal transactions: multivariate normal clusters
    X = rng.normal(loc=0.0, scale=1.0, size=(n, n_features))
    # Add some structure: make first 5 features correlated
    for i in range(1,5):
        X[:, i] += 0.5 * X[:, 0]
    # inject anomalies by adding large offsets to a small fraction
    n_anom = max(1, int(n * anomaly_frac))
    anom_idx = rng.choice(n, n_anom, replace=False)
    X[anom_idx] += rng.normal(loc=10.0, scale=4.0, size=(n_anom, n_features))
    df = pd.DataFrame(X, columns=[f"feat_{i+1}" for i in range(n_features)])
    df['is_synthetic_anomaly'] = 0
    df.loc[anom_idx, 'is_synthetic_anomaly'] = 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {{out_path}} with {{n}} rows, including {{n_anom}} injected anomalies.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--features', type=int, default=20)
    parser.add_argument('--frac', type=float, default=0.02)
    parser.add_argument('--out', type=str, default='data/transactions.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate(n=args.n, n_features=args.features, anomaly_frac=args.frac, random_state=args.seed, out_path=args.out)
