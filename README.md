**Overview**

This project implements an end-to-end unsupervised anomaly detection pipeline for high-dimensional transaction data.
It combines Isolation Forest, One-Class SVM, and PCA to identify anomalous patterns, analyze anomaly score distributions, and study the precision–recall tradeoff under different thresholds.
The pipeline is fully reproducible and designed to resemble a real-world anomaly scoring workflow used in financial and risk-monitoring systems.

**Key Objectives**

- Detect anomalous transactions without labeled training data
- Compare multiple unsupervised anomaly detection models
- Visualize high-dimensional data using PCA
- Analyze threshold-based precision/recall tradeoffs
- Save trained models and artifacts for reuse or deployment

**Tech Stack**
- Python
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- Jupyter Notebook

**Project Structure**
/data
  └── transactions.csv          # Synthetic transaction dataset
/notebooks
  └── Anomaly_Detection_PCA.ipynb  # Step-by-step notebook walkthrough
/src
  ├── generate_data.py           # Synthetic data generation
  └── train_and_score.py         # Model training, scoring, and visualization
/outputs                         # Saved models, reports, and visuals
README.md
requirements.txt

**Dataset Description**

The dataset is synthetically generated to simulate financial transaction data with injected anomalies.
- Each transaction contains multiple numerical features
- Normal transactions follow correlated multivariate distributions
- Anomalies are injected by adding large offsets to a small subset of samples

**Features**

- feat_1 … feat_n: Numerical transaction features

- is_synthetic_anomaly:
  0 = normal transaction
  1 = injected anomaly (used only for evaluation)

**Methodology**
***1. Data Generation***
Synthetic data is generated using multivariate normal distributions.
To introduce realistic structure:
- The first five features are correlated
- A small fraction of samples are selected as anomalies
- Anomalies are injected by adding large random offsets

***2. Preprocessing***
- Features are standardized using StandardScaler
- No labels are used during model training (unsupervised setting)

***3. Dimensionality Reduction***
- PCA is applied to reduce data to 2 dimensions
- PCA is used only for visualization, not for anomaly detection

***4. Anomaly Detection Models***
- Isolation Forest
  Learns isolation patterns in feature space
  Produces continuous anomaly scores

- One-Class SVM
  Learns a decision boundary around normal data
  Used for comparison with Isolation Forest

Higher scores indicate higher anomaly likelihood.

***5. Threshold Tuning & Evaluation***
- Anomaly thresholds are set using score percentiles (90, 95, 97.5, 99)
- Precision, Recall, and F1-score are computed at each threshold
- Results are saved as a JSON report for analysis

**Visualizations**

The pipeline automatically generates and saves:
- PCA 2D Scatter Plot
  Colored by Isolation Forest anomaly scores
  “outputs/pca_2d_scatter.png”

- Anomaly Score Distribution
  Histogram of Isolation Forest scores
  “outputs/anomaly_score_distribution.png”

**How to Run**
***1. Create Environment and Install Dependencies***
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

***2. Generate Synthetic Dataset (Optional)***
The dataset is already included.
To regenerate it with custom settings:

python src/generate_data.py \
  --n 10000 \
  --features 20 \
  --frac 0.02 \
  --out data/transactions.csv \
  --seed 42

***3. Train Models and Score Anomalies***
python src/train_and_score.py \
  --data_path data/transactions.csv \
  --out_dir outputs \
  --seed 42
