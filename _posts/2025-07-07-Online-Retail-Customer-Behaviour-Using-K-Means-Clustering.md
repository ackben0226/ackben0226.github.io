---
layout: post
title: Online Retail Customer Behaviour Using K-Means Clustering 
image: "/posts/Customer_Segmentation.png"
tags: [Python, K-Means]
---
# Customer Segmentation for Retail Growth | K-Means Clustering

**Driving 360° Customer Understanding Through Machine Learning**  
In this project we are going to leverage RFM analysis and K-Means Clustering to optimize marketing ROI for a retail growth.
The goal is to uncover hidden patterns in customer behavior and derive actionable business insights to help drive targeted marketing, customer retention, and sales growth strategies.

## 🎯 Business Impact Highlights
- **27% Revenue Concentration** in High-Value Cluster (5% of customer base)
- **Identified 18% At-Risk Customers** with Reactivation Potential
- **Built Dynamic Segmentation Framework** Reducing Campaign Costs by 32% (Simulated)
- **Enabled Personalized Marketing** Through 4 Distinct Behavioral Profiles

## Action

---
## 🔍 Technical Implementation
### 📊 Data Pipeline Architecture

## 🔧 Core Components
### 1. Data Preparation
   - Processed 541,909 transactions from UK retailer
   - Handled 24.9% missing CustomerIDs
   - Detected & treated outliers using IQR ranges
     
### 2. RFM Feature Engineering
We will create features that form the basis of RFM analysis (Recency, Frequency, Monetary) and Total Revenue, and are needed for K-Means clustering and business decision-making.
```ruby
# Creating column for total sale of each line
cleaned_data['TotalSale'] = cleaned_data['Quantity']*cleaned_data['UnitPrice']
```

```python
# Calculate Frequency and Monetary Values of customers
agg_data = cleaned_data.groupby(by = 'CustomerID', as_index = False) \
    .agg(MonetaryValue = ('TotalSale', 'sum'),           # Monetary
        Frequency = ('InvoiceNo', 'nunique'),            # Frequency
        LastInvoiceDate = ('InvoiceDate', 'max'))        

# Calculate Recency of Customers
max_invoice_date = agg_data['LastInvoiceDate'].max()
agg_data['Recency'] = (max_invoice_date - agg_data['LastInvoiceDate']).dt.days
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


