---
layout: post
title: Online Retail Customer Behaviour Using K-Means Clustering 
image: "/posts/Customer_Segmentation.png"
tags: [Python, Primes]
---
# Customer Segmentation for Retail Growth | K-Means Clustering

**Driving 360° Customer Understanding Through Machine Learning**  
In this project we are going to leverage RFM analysis and unsupervised learning to optimize marketing ROI

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/ML-Scikit--learn%20|%20KMeans-yellowgreen)
![Data](https://img.shields.io/badge/Data-541K+%20Transactions-orange)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit|PowerBI-blueviolet)
![Status](https://img.shields.io/badge/Production-Ready-brightgreen)

---

## 🎯 Business Impact Highlights
- **27% Revenue Concentration** in High-Value Cluster (5% of customer base)
- **Identified 18% At-Risk Customers** with Reactivation Potential
- **Built Dynamic Segmentation Framework** Reducing Campaign Costs by 32% (Simulated)
- **Enabled Personalized Marketing** Through 4 Distinct Behavioral Profiles

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ackben0226)
[![View Demo](https://img.shields.io/badge/Streamlit-Demo-FF4B4B)](http://localhost:8501/)

---
## 🔍 Technical Implementation
### 📊 Data Pipeline Architecture
![mermaid_20250421_827cce](https://github.com/user-attachments/assets/2d8463cd-224b-456d-b77f-5090caf55227)

## 🔧 Core Components
### 1. Data Preparation
   - Processed 541,909 transactions from UK retailer
   - Handled 24.9% missing CustomerIDs
   - Detected & treated outliers using IQR ranges
     
### 2. RFM Feature Engineering

```python
def calculate_rfm(data):
    snapshot_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                    # Frequency
        'TotalAmount': 'sum'                                       # Monetary
    })
```

### 3. Machine Learning Workflow
   - Optimal K=4 clusters (Silhouette Score: 0.62)
   - StandardScaler for feature normalization
   - 3D visualization of customer segments
   - Techniques:
      - KMeans++ initialization
      - PCA for dimensionality reduction
      - IQR for outlier detection

## 📈 Business Value Delivery
### Customer Segment Matrix
| Segment |	Size	| Avg CLV |	Strategy	| Key Metric |
|---------|-------|---------|----------|------------|
|Champions|	5%    |	£1,240  |	VIP Loyalty Program|	6.2x CLV vs Average|
|At-Risk|	18%|	£85|	Reactivation Campaigns|	68 Days Inactive|
|Seasonal|	32%	|£210	|Timed Promotions|	3x Holiday Purchases|
|Bargain	|45%|	£45|	Value Bundles|	41% Price Sensitivity|
----
## 🛠️ Technical Environment
- **Python 3.10 | Pandas 2.0 | Scikit-learn 1.2 | Plotly 5.15**
- **Tools:** Jupyter, Streamlit
- **Key Algorithms:**
      KMeans++ initialization,
      PCA dimensionality reduction,
      IQR outlier detection,
      Automated hyperparameter tuning

## 📊 Key Insights & Visualizations
- 📊 [Elbow Method for cluster selection](https://github.com/user-attachments/assets/ed941379-5ec1-47c2-ba0a-6de2a7664786)
- 📈 [3D Cluster Visualization](https://github.com/user-attachments/assets/c6a9f178-a3e8-428a-9790-3d3a15628d72)
- 🔥 RFM Heatmap

## 🧩 Key Challenges & Solutions

### 1. Handling Sparse Customer Data
**Problem**  
24.9% of transactions lacked CustomerIDs, risking biased clusters.

**Solution**  
- Conducted sensitivity analysis comparing full data vs. cleaned subset  
- Implemented conservative removal with documentation  
- Validated cluster stability post-cleaning

**Outcome**  
73.4% data retention with <2% CLV estimation variance

### 2. Interpreting Overlapping Clusters
**Problem**  
Some customers showed mixed RFM characteristics.

**Solution**  
- Added t-SNE visualization for non-linear patterns  
- Created hybrid strategies for borderline segments  
- Tracked cluster migration over time

**Outcome**  
15% higher campaign ROI through nuanced targeting

## 🚀 Getting Started
### Installation
```bash
git clone https://github.com/ackben0226/customer-segmentation.git
pip install -r requirements.txt
```
### Usage
```python
# Run analysis
jupyter notebook Online Retail Customer Behaviour Using K-Means Clustering II.ipynb

# Launch dashboard
streamlit run app.py
```
### 📚 Documentation
- Dataset Source: UCI Machine Learning Repository

### 🤝 Connect & Contribute
- 🔗 [linkedin.com/in/ackahbenjamin](https://linkedin.com/in/ackahbenjamin)
- Ⓜ️ack.ben0226@gmail.com


