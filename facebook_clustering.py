import pandas as pd
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('C:/ml/python/data/kmeans_fb.csv',delimiter=',')
data.info()
data.isna().sum()
data = data.drop(['Column1','Column2','Column3','Column4','status_id','status_published'],axis=1)

df = data.copy()
df.info()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['status_type'] = le.fit_transform(df['status_type'])





from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

df_scaled = ms.fit_transform(df)

from sklearn.cluster import KMeans

# storing iteration of 10 clusters iterated 10 times to find labels and intertia

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=20,max_iter=10,random_state=0)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

kmeans.inertia_
kmeans.labels_


#Elbow Method for optimal number of clusters

import matplotlib.pyplot as plt
plt.plot(range(1,11),inertia,marker='*')
plt.xlabel('Clusters')
plt.ylabel('inertia')
plt.title('Elbow Method')

# Optimal Number of Clusters
k = 4

kmeans = KMeans(n_clusters=k,init='k-means++',n_init=20,max_iter=10,random_state=0)
kmeans.fit(df_scaled)
labels = kmeans.labels_
df['cluster'] = labels

plt.scatter(df_scaled[:,0],df_scaled[:,1],c=df['cluster'],cmap='viridis',s=300,linewidths=10,edgecolors='r')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='x',s=300,linewidth=10,color='k')
plt.show()

print(kmeans.inertia_)

# scoring the cluster at k = 4
from sklearn.metrics import silhouette_score

SilhouetteScore = silhouette_score(df_scaled,labels)
print(f'score: {SilhouetteScore*100:.2f}%')


