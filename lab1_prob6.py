import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# loading raw data #

mydata = pd.read_csv('ltemaster.csv', encoding='unicode_escape')
ltedata = mydata.dropna()

# encoding categorical variables #

le = LabelEncoder()
ltedata.Wind = le.fit_transform(ltedata.Wind)
ltedata.Condition = le.fit_transform(ltedata.Condition)

#  Checking the correlation #

corr = (ltedata.corr())
print("top 10 co-related features are below:")
print(corr['sigl'].sort_values(ascending=False)[1:11])  # top 10 co-related features

print("Please enter the type of clustering: Enter 1 for all variable, else for top 7 co-related features")
p = int(input())

if p != 1:
    ltedata = ltedata[['ss', 'signal', 'asu', 'rsrq', 'rssnr', 'alt', 'Condition']]
scaler = StandardScaler()
scaler.fit(ltedata)
x_scaler = scaler.transform(ltedata)

# Obtaining optimum number of cluster from Elbow plot#

mylist = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, random_state=123)
    kmeans.fit(x_scaler)
    mylist.append(kmeans.inertia_)

plt.plot(range(1, 11), mylist)
plt.title('The graph for elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# ### building the kmeans model ###


km_2 = KMeans(n_clusters=2, random_state=123)
km_2.fit(x_scaler)
y_cluster_kmeans = km_2.predict(x_scaler)
score = metrics.silhouette_score(x_scaler, y_cluster_kmeans)
if p == 1:
    print("silhouette score for K = 2 using all features is --> " + str(score))
else:
    print("silhouette score for K = 2 using top 7 co-related features is --> " + str(score))

km_3 = KMeans(n_clusters=3, random_state=123)
km_3.fit(x_scaler)
y_cluster_kmeans = km_3.predict(x_scaler)
score = metrics.silhouette_score(x_scaler, y_cluster_kmeans)
if p == 1:
    print("silhouette score for K = 3 using all features is --> " + str(score))
else:
    print("silhouette score for K = 3 using top 7 co-related features is --> " + str(score))