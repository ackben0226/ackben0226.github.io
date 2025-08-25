---
Post:
title: 'Enhancing Cybersecurity Resilience: Leveraging Machine Learning for Advanced Threat Detection and Response'
image: "/posts/cyberimage.jpg"
tag: 
---
This project explores the application of machine learning techniques to cybersecurity, focusing on network traffic classification for intrusion detection. It compares supervised learning models like Random Forest (RF), XGBoost (XGB), Decision Trees (DT), and Support Vector Machines (SVM) across benchmark datasets such as CICIDS2017 and UNSW-NB15. The research aims to enhance threat detection accuracy, minimize false positives, and optimize resource use in cybersecurity operations.

## Table of Content
- Project Description
- Actions

## Actions
We extracted the CICIDS2017 benchmark dataset, which contains multiple network traffic files, including DDoS, PortScan, Infiltration, Web Attacks, and regular working-hour traffic. To ensure balanced and manageable processing, we randomly sampled 31,437 instances from each file, saved them as CSV files, before concatenating them into a unified dataset.

```python
dfs = []
for file in files:
    df = pd.read_csv(os.path.join(file_path, file))
    dfs.append(df.sample(n=31437, random_state=1))
merged_data = pd.concat(dfs, ignore_index=True)  # Concatenate files
```
### Data Preprocessing
After successfully concatenating the files, we standardized column names by stripping whitespace, cleaned categorical features by removing corrupted symbols (�), and replaced infinite values with NaN and dropped incomplete rows. This ensures a clean, reliable dataset ready for modeling.

```python
merged_data.columns = merged_data.columns.str.strip()                # Strip whitespace
for col in merged_data.columns:
    if(merged_data[col].dtype == "object"):
        merged_data[col] = merged_data[col].str.replace('�', '')    # Replace unknown symbols/characters
merged_data.replace([np.inf, -np.inf], np.nan, inplace=True)         # Replace infinite values with NaN
merged_data = merged_data.dropna(how='any')                          # Drop missing values
```

### Feature Scaling & Multicollinearity Check
To prepare the dataset for machine learning models, we first standardized the features using `StandardScaler`. This ensures that all features have zero mean and unit variance and are on the same scale.
<br/> Next, we evaluated multicollinearity using the Variance Inflation Factor (VIF). High VIF values indicate that a feature is highly correlated with others, which can degrade model performance. Features with VIF > 10 were flagged for removal to reduce redundancy and improve model stability.

```ruby
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Initialize 
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Compute VIF for each feature
vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
vif['features'] = X_scaled.columns

# Identify highly collinear features
features_to_remove = list(vif.loc[vif['vif_factor'] > 10, 'features'])
print(features_to_remove)
```
### Label Encoding & Final Feature Preparation
Machine learning models cannot work directly with categerical target variables. In this case, the labels (benign/malicious) were encoded into numerical values using LabelEncoder, so they can be used by supervised learning models.
<br/>We then prepared the feature matrix (X) and target vector (y) for model training.
This step ensures that the dataset is ready for model training with all features scaled and target labels numeric.
```ruby
from sklearn.preprocessing import LabelEncoder

# Encode target labels
lab_enc = LabelEncoder()
merged_data['Label'] = lab_enc.fit_transform(merged_data['Label'])

# Prepare features and labels
feats = X_scaled2          # preprocessed and scaled feature set
label = merged_data['Label'].values
```

### Feature Importance Analysis with Random Forests
In this section, I conducted a feature importance analysis to understand which attributes most strongly influenced model predictions. an essential step for explainability in security operations As part of my cybersecurity threat detection research. This step was essential for improving explainability, a critical requirement in security operations where practitioners need to justify and trust automated decisions.
<br/> I trained the dataset using `Random Forest Regressor (n_estimators = 100, max_depth = 10)` model. This model assigned an importance score to each feature, quantifying its contribution to prediction accuracy.
<br/> I then visualized the most influential features in ascending order of importance using __Matplotlib library__ as shown in the horizontal barplot below.

```python
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 100, max_depth = 10)
forest.fit(train_data, train_labels)
feature_names = feats.columns
importances = forest.feature_importances_
print('feature importance:', importances)

import matplotlib.pyplot as plt

feature_importance = feature_importance.sort_values('importance', ascending=True)

# Plotting
plt.figure(figsize=(8, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
plt.xlabel('Importances')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.show()
```
<img width="844" height="547" alt="image" src="https://github.com/user-attachments/assets/9fb941fa-aaa8-451a-b8ec-e75c1d7b1c46" />

### PCA for Dimension Reduction
After encoding the categorical features, the dataset became highly dimensional. To address this, I applied Principal Component Analysis (PCA), reducing the feature space to 10 principal components while retaining the most meaningful patterns. This approach preserved the features with the greatest variance and predictive power, improved computational efficiency, and reduced noise, ultimately making the modeling process more robust and easier to interpret.
```ruby
from sklearn.decomposition import PCA

# Reducing dimensions to 10 components
pca = PCA(n_components=10)
merged_data_pca = pca.fit_transform(X_scaled2)

# Transform pca into DataFrame of 10 dimensions
merged_data_pca_df = pd.DataFrame(data=merged_data_pca, columns=[f'PC{i+1}' for i in range(10)])
merged_data_pca_df.head()
```

| Index | PC1       | PC2       | PC3       | PC4       | PC5       | PC6       | PC7       | PC8       | PC9       | PC10      |
| ----- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| 0     | 2.656354  | 0.407357  | 0.069625  | -0.267616 | -0.097811 | -0.343545 | 0.010062  | 0.143420  | 0.378717  | -0.221853 |
| 1     | 1.786847  | 0.205950  | -0.013894 | 0.002700  | -0.028225 | -0.239581 | 0.184643  | 0.337759  | 0.121639  | 0.013272  |
| 2     | 2.341815  | 0.343219  | 0.053673  | -0.199320 | -0.086107 | -0.226751 | -0.005145 | 0.118177  | 0.215375  | -0.111739 |
| 3     | -0.700773 | 0.461455  | -1.589482 | -0.353896 | -0.298374 | 0.190028  | 0.383688  | -0.556352 | -0.415059 | 0.882087  |
| 4     | -0.529991 | -1.469167 | 0.657939  | 0.207449  | 0.064370  | 0.557679  | 0.049590  | 0.536566  | -0.655648 | 0.032474  |

<img width="500" height="393" alt="image" src="https://github.com/user-attachments/assets/d6afc719-c428-4ddd-9635-70034156c61e" />

### Splitting Data into Training and Testing Sets
To evaluate model performance objectively, we split the dataset into `training` and `testin`g sets into a 70:30 ratio using `train_test_split` from scikit-learn after the PCA. This ensures that the training set provides enough data for model learning while reserving a test set for unbiased evaluation. A fixed random seed was used to maintain reproducibility.

```ruby
import numpy as np
from sklearn.model_selection import train_test_split

# feats and label
feats = merged_data_pca_df
label = merged_data['Label'].values

# Generate IDs for splitting
all_ids = numpy.arange(0, feats.shape[0])

random_seed = 1

# Then splitting the data 70:30 into training and test sets
train_set_ids, test_set_ids = train_test_split(all_ids, test_size=0.3, train_size=0.7,
                                              random_state=random_seed, shuffle=True)

# Training set
X_train = feats.iloc[train_set_ids, :]
y_train = label[train_set_ids]

# Testing set
X_test = feats.iloc[test_set_ids, :]
y_test = label[test_set_ids]
```
### Model Training for SVC, RF, XGB and DT
After preprocessing, dimensionality reduction (PCA) and splitting the dataset, I trained multiple supervised learning models to evaluate their performance on the dataset. The following models were implemented:
__Support Vector Classifier (SVC)__,__Random Forest (RF)__, __XGBoost (XGB)__, __Decision Tree (DT)__

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize models
svc = SVC()
rfc = RandomForestClassifier(n_estimators= 100, max_depth = 10)
xgb = XGBClassifier()
dt_model = DecisionTreeClassifier()

# Fit models
svc_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
```
### Model Prediction
After I had trained and fitted each model, I used them to make predictions on the test set. This step allowed me to evaluate how well the models generalize to unseen data and assess their predictive performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. 

```ruby
# Make predictions
y_pred_svc = svc_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

# Predict probabilities for ROC-AUC
y_prob_svc = svc_model.predict_proba(X_test)[:, 1]
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
```
## Result
__Performance Analysis of each model__
The table below summarizes the performance of the four trained models on the test set:
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| ----- | ------------ | ------------- | ---------- | ------------ 
|
| SVM   | 90.48        | 95.86         | 94.05      | 94.95        |
| RF    | 98.44        | 99.22         | 98.91      | 99.07        |
| DT    | 98.38        | 98.99         | 99.08      | 99.03        |
| XGB   | 98.52        | 99.29         | 98.97      | 99.13        
|

__Key Insights:__
- All models achieved high predictive performance, with ensemble methods (Random Forest and XGBoost) slightly outperforming the single-tree and SVM models.
- XGBoost achieved the highest F1-score, indicating excellent balance between precision and recall.
- SVM performed well but lagged slightly behind ensemble methods, likely due to high-dimensional data despite PCA.

<img width="750" height="350" alt="image" src="https://github.com/user-attachments/assets/52a9f553-756d-41b2-bbe8-cb15156eb26d" />


