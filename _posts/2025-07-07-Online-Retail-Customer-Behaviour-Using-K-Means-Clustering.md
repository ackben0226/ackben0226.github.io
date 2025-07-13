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

## Action
To successfully embark on this project, it is essential to first ensure that the data we work with is free from irregularities, such as missing values, outliers, and other inconsistencies.

## Missing Values
The CustomerID column in the dataset has approximately 25% missing values, which is substantial. Rather than dropping these rows, we opt to impute the missing values using the median.

```ruby
# impute rows where values are missing
median_value = data['CustomerID'].median()
data['CustomerID'].fillna(median_value, inplace=True)
```

### Outlier
The presence of outliers can distord/shift the cluster centroids and lead to less accurate cluster assignments. This significantly affects the effectiveness of K-means clustering.   
To handle outliers, careful consideration is needed since extreme values may represent important customer behaviors rather than errors.
<br/> In this section, we use Pandas’ .describe() method to examine the distribution and spread of our RFM variables — Recency, Frequency, and Monetary Value. 
<br/> The summary statistics obtained help us understand the data characteristics before proceeding with clustering. The results are shown in the table below.

|   Metric  | Quantity |     InvoiceDate     |  UnitPrice | CustomerID |
| :-------: | :------: | :-----------------: | :--------: | :--------: |
|   count   |  541,909 |       541,909       |   541,909  |   406,829  |
|    mean   |   9.55   | 2011-07-04 13:34:57 |    4.61    |  15,287.69 |
|    min    |  -80,995 | 2010-12-01 08:26:00 | -11,062.06 |   12,346   |
|    25%    |   1.00   | 2011-03-28 11:34:00 |    1.25    |   13,953   |
|    50%    |   3.00   | 2011-07-19 17:17:00 |    2.08    |   15,152   |
|    75%    |   10.00  | 2011-10-19 11:27:00 |    4.13    |   16,791   |
|    max    |  80,995  | 2011-12-09 12:50:00 |  38,970.00 |   18,287   |
|    std    |  218.08  |         N/A         |    96.76   |  1,713.60  |

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

## Dealing with Outliers
From the summary statistics above, max column values of quantity and unit price are much higher than the median value, depicting outliers.
Because of this, we apply some outlier removal in order to facilitate generalisation across the full dataset.
<br/> We apply the 'boxplot method' by removing any rows where the values in the RFM columns fall outside twice the interquartile range (IQR).

```ruby
# Dealing with Monetary Outlier
mon_q1 = agg_data['MonetaryValue'].quantile(0.25)
mon_q3 = agg_data['MonetaryValue'].quantile(0.75)

IQR = mon_q3 - mon_q1

monetary_outliers = agg_data[(agg_data['MonetaryValue'] > (mon_q3 + 1.5 * IQR))
    | (agg_data['MonetaryValue'] < (mon_q1 - 1.5 * IQR))].copy()

# Dealing with Frequency Outlier
freq_q1 = agg_data['Frequency'].quantile(0.25)
freq_q3 = agg_data['Frequency'].quantile(0.75)

IQR = freq_q3 - freq_q1

frequency_outliers = agg_data[(agg_data['Frequency'] > (freq_q3 + 1.5 * IQR))
    | (agg_data['Frequency'] < (freq_q1 - 1.5 * IQR))].copy()

# Dealing with Recency Outlier
rec_q1 = agg_data['Recency'].quantile(0.25)
rec_q3 = agg_data['Recency'].quantile(0.75)

IQR = rec_q3 - rec_q1

recency_outliers = agg_data[(agg_data['Recency'] > (rec_q3 + 1.5 * IQR))
    | (agg_data['Recency'] < (rec_q1 - 1.5 * IQR))].copy()
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

[!image][<img width="1014" height="547" alt="image" src="https://github.com/user-attachments/assets/1c00101a-37d5-4947-8be2-3c87e2d32454" />
]

### Cluster Label
To identify customers such as high-value loyal customers, at-risk customers, and one-time buyers, K-Means clustering uses RFM scores (numerical values) to group customers into clusters and, hence labeled as either **re-engage, retain, delight, reward, etc.** based on similarities in their purchasing behavior. Each customer is assigned to the cluster whose centroid best represents their RFM profile.
We visualize a 3-dimensional scatter plot for the clusters to provide a better understanding of these segments, with Recency, Frequency, and Monetary value on each axis.

[!image][<img width="794" height="812" alt="image" src="https://github.com/user-attachments/assets/5636b4c4-c9bd-4545-ab96-de564e1b2e6d" />
]

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


## 📈 Business Value Delivery

### Customer Segment Matrix

| **Cluster**   | **Label**    | **Characteristics**                                                              | **Actionable Strategy**                                                                 |
|---------------|--------------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Cluster 0     | Retain       | High-spending, moderately frequent buyers; loyal but not very recent.             | Implement loyalty programs, personalized offers, and regular engagement to ensure they remain active.                            |
| Cluster 1     | Re-Engage    | Infrequent, low-value buyers with long recency; disengaged.                       | Use targeted marketing campaigns, special discounts, or reminders to encourage them to return and purchase again.                  |
| Cluster 2     | Nurture      | New or passive customers with low spending but recent activity.                   | Focus on building relationships, providing excellent customer service, and offering incentives to encourage more frequent purchases.                        |
| Cluster 3     | Reward       | High-frequency, high-value, recent customers—our most loyal group.                | Implement a robust loyalty program, provide exclusive offers, and recognize their loyalty to keep them engaged and satisfied.      |
| Cluster -1    | Pamper       | Big spenders with infrequent purchases; high R, low F.                            | Focus on maintaining their loyalty with personalized offers or luxury services that cater to their high spending capacity.            |
| Cluster -2    | Upsell       | Frequent shoppers with low average spend.                                         | Implement loyalty programs or bundle deals to encourage higher spending per visit, given their frequent engagement.             |
| Cluster -3    | Delight      | Top-tier customers—extreme in recency, frequency, and monetary value.             |  Develop VIP programs or exclusive offers to maintain their loyalty and encourage continued engagement..          |




## Growth & Next Steps
While RFM and K-Means yielded meaningful segments, using DBSCAN or Gaussian Mixture Models algorithms could capture more complex patterns. Further tuning of K-Means parameters and ensemble methods may improve cluster stability.

Incorporating additional data such as demographics and engagement metrics, along with advanced feature engineering on purchase timing and product preferences, would enhance segmentation and enable more targeted marketing strategies.

### 📚 Documentation
- Dataset Source: UCI Machine Learning Repository

### 🤝 Connect & Contribute
- 🔗 [linkedin.com/in/ackahbenjamin](https://linkedin.com/in/ackahbenjamin)
- Ⓜ️ack.ben0226@gmail.com


