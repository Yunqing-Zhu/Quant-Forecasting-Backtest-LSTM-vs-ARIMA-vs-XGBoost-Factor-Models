
import argparse, os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['is_synthetic_anomaly']).values
    y = df['is_synthetic_anomaly'].values
    return X, y, df

def fit_models(X_train, random_state=42):
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    pca = PCA(n_components=2, random_state=random_state).fit(Xs)
    X_pca = pca.transform(Xs)

    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=random_state)
    iso.fit(Xs)
    iso_scores = -iso.score_samples(Xs)  # higher -> more anomalous

    oc = OneClassSVM(nu=0.02, kernel='rbf', gamma='scale')
    oc.fit(Xs)
    oc_scores = -oc.decision_function(Xs)  # higher -> more anomalous

    return scaler, pca, iso, oc, Xs, X_pca, iso_scores, oc_scores

def save_visuals(X_pca, iso_scores, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=iso_scores, alpha=0.6)
    plt.title('PCA 2D scatter (colored by IsolationForest score)')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.colorbar(label='anomaly score')
    pca_path = os.path.join(out_dir, 'pca_2d_scatter.png')
    plt.savefig(pca_path, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.hist(iso_scores, bins=100)
    plt.title('IsolationForest anomaly score distribution')
    plt.xlabel('anomaly score'); plt.ylabel('count')
    hist_path = os.path.join(out_dir, 'anomaly_score_distribution.png')
    plt.savefig(hist_path, bbox_inches='tight')
    plt.close()
    print(f'Saved visuals to {{out_dir}}')

def threshold_and_report(scores, y_true, thresholds=None):
    if thresholds is None:
        thresholds = np.percentile(scores, [90,95,97.5,99])
    reports = {}
    for t in thresholds:
        preds = (scores >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        reports[float(t)] = {'precision':float(p),'recall':float(r),'f1':float(f)}
    return reports

def main(args):
    X, y, df = load_data(args.data_path)
    scaler, pca, iso, oc, Xs, X_pca, iso_scores, oc_scores = fit_models(X, random_state=args.seed)
    # save models
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(args.out_dir, 'pca.joblib'))
    joblib.dump(iso, os.path.join(args.out_dir, 'isolation_forest.joblib'))
    joblib.dump(oc, os.path.join(args.out_dir, 'oneclass_svm.joblib'))

    # visuals
    save_visuals(X_pca, iso_scores, args.out_dir)

    # threshold reports
    reports = threshold_and_report(iso_scores, y)
    report_path = os.path.join(args.out_dir, 'threshold_reports.json')
    import json
    with open(report_path, 'w') as f:
        json.dump(reports, f, indent=2)
    print('Saved threshold reports.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/transactions.csv')
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
