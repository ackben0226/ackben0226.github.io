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
import os
import pandas as pd
import numpy as np

file_path = "/content/drive/MyDrive/Colab Notebooks"
files = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

dfs = []

for file in files:
    file_path_full = os.path.join(file_path, file)
    df = pd.read_csv(file_path_full)
    selected_df = df.sample(n = 31437, random_state = 1)
    dfs.append(selected_df)

# Save each selected DataFrame to a new CSV file
for i, selected_df in enumerate(dfs):
    new_file_path = os.path.join(file_path, f"selected_{files[i]}")
    selected_df.to_csv(new_file_path, index=False)

print("Processed all datasets and saved the selected rows to new files.")

```
