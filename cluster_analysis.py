"""
Week 5 Task: Cluster Analysis and Market Segmentation in Automotive Data
Description: Applies K-Means clustering and Silhouette analysis to segment automotive market data.
Author: Suyash Vasant Sonawane
Repository: https://github.com/suyash-6011/yuva-intern-week5-cluster-analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Load or Simulate Automotive Dataset
# (Replace with your actual dataset file path if available, e.g., pd.read_csv('automotive_data.csv'))
np.random.seed(42)
n_samples = 300

data = {
    'Price_USD': np.random.randint(15000, 85000, n_samples),
    'Horsepower': np.random.randint(80, 450, n_samples),
    'Engine_Size_L': np.round(np.random.uniform(1.0, 6.0, n_samples), 1),
    'Fuel_Efficiency_MPG': np.random.randint(15, 50, n_samples),
}

df = pd.DataFrame(data)
print('--- Dataset Sample ---')
print(df.head())

# 2. Data Preprocessing & Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_features, columns=df.columns)

# 3. Determining Optimal Clusters (Elbow Method & Silhouette Scores)
inertia = []
silhouette_scores = []
K_range = range(2, 7)

for k in K_range:
  kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
  kmeans.fit(df_scaled)
  inertia.append(kmeans.inertia_)
  score = silhouette_score(df_scaled, kmeans.labels_)
  silhouette_scores.append(score)
  print(f'Clusters K={k} | Silhouette Score: {score:.4f}')

# 4. Apply Final K-Means Model (Optimal K = 3 based on standard market tiers)
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(df_scaled)

print('\n--- Cluster Distribution ---')
print(df['Cluster'].value_counts())

# 5. Visualization & Saving Plots
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Price_USD',
    y='Horsepower',
    hue='Cluster',
    data=df,
    palette='viridis',
    s=70,
)
plt.title('Automotive Market Segmentation - K-Means Clustering')
plt.xlabel('Vehicle Price (USD)')
plt.ylabel('Horsepower')
plt.legend(title='Market Cluster')
plt.savefig('cluster_visualization.png')
print(
    '\nPlot saved successfully as cluster_visualization.png on your workspace!'
)
