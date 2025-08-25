---
Post:
image:
title: 'Enhancing Cybersecurity Resilience: Leveraging Machine Learning for Advanced
Threat Detection and Response'
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
### Splitting Data into Training and Testing Sets
To evaluate model performance objectively, we split the dataset into `training` and `testin`g sets into a 70:30 ratio using `train_test_split` from scikit-learn. This ensures that the training set provides enough data for model learning while reserving a test set for unbiased evaluation. A fixed random seed was used to maintain reproducibility.

```ruby
import numpy as np
from sklearn.model_selection import train_test_split

# Generate IDs for splitting
all_ids = np.arange(0, feats.shape[0])
random_seed = 1

# Splitting 70:30
train_set_ids, test_set_ids = train_test_split(
    all_ids, test_size=0.3, train_size=0.7,
    random_state=random_seed, shuffle=True
)
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
