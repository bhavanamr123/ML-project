# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:45:00 2023

@author: bhava
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv("C:\\Users\\bhava\\Desktop\\7th sem\\ML\\assignment\\data.csv", header=None)

x = df.iloc[:, 0]
y = df.iloc[:, 1]


plt.scatter(x, y)
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.title('Visualization of data points through Scatter Plot')
plt.show()


pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df)

num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(reduced_data)

df['Cluster'] = clusters


plt.scatter(x, y, c=clusters, cmap='viridis')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clustered Data')
plt.show()


df.to_csv('clustered_data.csv', index=False)



