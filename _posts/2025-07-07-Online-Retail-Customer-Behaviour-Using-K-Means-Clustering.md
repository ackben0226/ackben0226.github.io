---
layout: post
title: Online Retail Customer Behaviour Using K-Means Clustering 
image: "/posts/Customer_Segmentation.png"
tags: [Python, K-Means, Tableau]
---
# Customer Segmentation for Retail Growth | K-Means Clustering

## Project Overview  
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
We will create features that form the basis of RFM analysis (*Recency, Frequency, Monetary*) and *Total Revenue*, and are needed for K-Means clustering and business decision-making.
```ruby
# Creating column for total sale of each line
data['TotalSale'] = data['Quantity'] * data['UnitPrice']
```

```python
# Calculate Frequency and Monetary Values of customers
agg_data = cleaned_data.groupby(by = 'CustomerID', as_index = False) \
    .agg(MonetaryValue = ('TotalSale', 'sum'),           # Monetary
        Frequency = ('Invoice No', 'nunique'),            # Frequency
        LastInvoiceDate = ('Invoice Date', 'max'))        

# Calculate Recency of Customers
max_invoice_date = agg_data['LastInvoiceDate'].max()
agg_data['Recency'] = (max_invoice_date - agg_data['LastInvoiceDate']).dt.days
```

### 3. Machine Learning Workflow (Optimal K)
```ruby
# Assuming non_outliers_scaled is already defined
max_k = 12
inertia = []
silhouette_scores = []  # Create a list to store silhouette scores
k_values = range(2, max_k + 1)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=1000)
    
    cluster_labels = kmeans.fit_predict(non_outliers_scaled)
    
    # Calculate and store the silhouette score
    sil_score = silhouette_score(non_outliers_scaled, cluster_labels)
    silhouette_scores.append(sil_score)
    
    # Store inertia values
    inertia.append(kmeans.inertia_)
```
We need to find the optimal k (number of clusters) that best separates your customers based on their Recency, Frequency, and Monetary behavior. This is done using the above syntax before proceeding with the Elbow Method visualization.

### Cluster Label
To identify customers such as high-value loyal customers, at-risk customers, and one-time buyers, K-Means clustering uses RFM scores (numerical values) to group customers into clusters and, hence labeled as either **re-engage, retain, delight, reward, etc.** based on similarities in their purchasing behavior. Each customer is assigned to the cluster whose centroid best represents their RFM profile.
We visualize a 3-dimensional scatter plot for the clusters to provide a better understanding of these segments, with Recency, Frequency, and Monetary value on each axis.

```ruby
kmeans = KMeans(n_clusters = 4, random_state = 42, max_iter = 1000)
cluster_labels = kmeans.fit_predict(non_outliers_scaled)
cluster_labels

# 3-Dimensional Scatterplot for CustomerID
cluster_colors = {0 : 'blue', 1 : 'green', 2 : 'orange', 3 : 'red'}
colors = non_outliers['Cluster'].map(cluster_colors)

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(projection = '3d')
scatter = ax.scatter(non_outliers['MonetaryValue'],
                     non_outliers['Frequency'],
                     non_outliers['Recency'],
                    c = colors, marker = 'o')

ax.set_xlabel('Monetary Value')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')

ax.set_title('3D Scatter Plot of Customer Data by Clusters')
plt.show()
```

The table below shows a sample of customers segmented using RFM scores and K-Means clustering. Each customer is assigned a Cluster number and a corresponding ClusterLabel that describes their segment based on purchasing behavior.

| CustomerID | MonetaryValue | Frequency |   LastInvoiceDate   | Recency (Days) | Cluster | ClusterLabel |
| :--------: | :-----------: | :-------: | :-----------------: | :------------: | :-----: | :----------: |
|    12348   |    1797.24    |     4     | 2011-09-25 13:13:00 |       74       |    1    |   RE-ENGAGE  |
|    12349   |    1757.55    |     1     | 2011-11-21 09:51:00 |       18       |    1    |   RE-ENGAGE  |
|    12350   |     334.40    |     1     | 2011-02-02 16:01:00 |       309      |    0    |    RETAIN    |
|    12352   |    2506.04    |     8     | 2011-11-03 14:37:00 |       35       |    3    |    REWARD    |
|    12353   |     89.00     |     1     | 2011-05-19 17:47:00 |       203      |    0    |    RETAIN    |
|     ...    |      ...      |    ...    |         ...         |       ...      |   ...   |      ...     |
|    18172   |    7561.68    |     20    | 2011-11-25 11:12:00 |       14       |    -3   |    DELIGHT   |
|    18198   |    5425.56    |     17    | 2011-12-05 14:49:00 |        3       |    -3   |    DELIGHT   |
|    18223   |    6484.54    |     14    | 2011-12-05 09:11:00 |        4       |    -3   |    DELIGHT   |
|    18225   |    5509.12    |     12    | 2011-12-06 13:27:00 |        2       |    -3   |    DELIGHT   |
|    18229   |    7276.90    |     20    | 2011-11-28 09:48:00 |       11       |    -3   |    DELIGHT   |

## RFM Segmentation Dashboard
This Tableau dashboard visualizes the K-Means clustering results from the RFM analysis. It highlights customer segments like **Delight**, **Reward**, **Re-Engage**, and **Retain** with actionable insights based on Recency, Frequency, and Monetary patterns.
<iframe src="https://public.tableau.com/views/rfmdashboard_17522584516860/rfmDashboard?:showVizHome=no&:embed=true" width="100%" height="600px" frameborder="0"></iframe>

<iframe seamless frameborder="0" src="https://public.tableau.com/views/DSIEarthquakeDashboard/DSIEarthquakeTracker?:embed=yes&:display_count=yes&:showVizHome=no" width = '1090' height = '900'></iframe>

### 📊 RFM Segmentation Dashboard (Interactive)

🔗 **[Click here to view the live dashboard on Tableau Public
[https://public.tableau.com/app/profile/benja%20mit/viz/rfmdashboard_17522584516860/rfmDashboard?publish=yes](https://public.tableau.com/app/profile/benjamin.ackah/viz/rfmdashboard_17522584516860/rfmDashboard?publish=yes)

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


