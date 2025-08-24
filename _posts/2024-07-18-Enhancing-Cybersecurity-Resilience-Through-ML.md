---
Post:
image:
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
## Data Preprocessing
After successfully concatenating the files, we standardized column names by stripping whitespace, cleaned categorical features by removing corrupted symbols (�), and replaced infinite values with NaN and dropped incomplete rows. This ensures a clean, reliable dataset ready for modeling.

```python
merged_data.columns = merged_data.columns.str.strip()                # Strip whitespace
for col in merged_data.columns:
    if(merged_data[col].dtype == "object"):
        merged_data[col] = merged_data[col].str.replace('�', '')    # Replace unknown symbols/characters
merged_data.replace([np.inf, -np.inf], np.nan, inplace=True)         # Replace infinite values with NaN
merged_data = merged_data.dropna(how='any')                          # Drop missing values
```
