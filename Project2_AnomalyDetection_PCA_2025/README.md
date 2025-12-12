# Unsupervised Learning for Anomaly Detection in Transaction Data (2025)

**Summary**
Developed an end-to-end anomaly detection pipeline using Isolation Forest, One-Class SVM, and PCA on synthetic high-dimensional financial transaction data.

**Tech Stack**
Python · scikit-learn · pandas · numpy · seaborn · matplotlib · joblib · jupyter

**Key Features**
- High-dimensional preprocessing
- PCA for dimensionality reduction
- Isolation Forest anomaly scoring
- OC-SVM comparison
- Threshold tuning for precision/recall tradeoff
- Deployable pipeline for anomaly scoring

**Visuals**
- PCA 2D scatter (saved to `visuals/pca_2d_scatter.png`)
- Anomaly score distribution (saved to `visuals/anomaly_score_distribution.png`)

**Structure**
```
/data                    # synthetic dataset (transactions.csv)
/notebooks               # Jupyter notebook demonstration
/src                     # python scripts (data generation, training, scoring)
/visuals                 # generated plots
README.md
requirements.txt
```
**How to run**
```bash
# create env and install
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# generate data (optional - dataset already included)
python src/generate_data.py

# run training & scoring (produces visuals and model files)
python src/train_and_score.py --data_path data/transactions.csv --out_dir outputs

# open the notebook for step-by-step exploration
jupyter notebook notebooks/Anomaly_Detection_PCA.ipynb
```

**Notes**
- The dataset included is synthetic and for demo purposes only.
- The notebook walks through preprocessing, PCA visualization, model training, and evaluation.
