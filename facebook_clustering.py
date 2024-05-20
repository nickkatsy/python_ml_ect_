import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/kmeans_fb.csv')

df = df.drop(['Column1','Column2','Column3','Column4','status_id','status_published'],axis=1)




from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['status_type'] = le.fit_transform(df['status_type'])





from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

df_scaled = ms.fit_transform(df)





from sklearn.cluster import KMeans



inertia = []
for i in range(1,12):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=20,max_iter=10,random_state=0).fit(df_scaled)
    inertia.append(kmeans.inertia_)

kmeans.inertia_
kmeans.labels_


#Elbow Method for optimal number of clusters

import matplotlib.pyplot as plt
plt.plot(range(1,12),inertia,marker='*')
plt.xlabel('Clusters')
plt.ylabel('inertia')
plt.title('Elbow Method')

# Optimal Number of Clusters k = 3

kmeans = KMeans(n_clusters=3,init='k-means++',n_init=20,max_iter=10,random_state=0).fit(df_scaled)
labels = kmeans.labels_
df['cluster 1'] = labels
df['cluster 2'] = labels


plt.scatter(df_scaled[:,0],df_scaled[:,1],c=df['cluster 1'],cmap='viridis',s=300,linewidths=10,edgecolors='r')
plt.scatter(df_scaled[:,0],df_scaled[:,1],c=df['cluster 2'],cmap='viridis',s=300,linewidths=10,edgecolors='m')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='x',s=300,linewidth=10,color='k')
plt.show()



# scoring the cluster at k = 3
from sklearn.metrics import silhouette_score

SilhouetteScore = silhouette_score(df_scaled,labels)
print(f'score: {SilhouetteScore*100:.2f}%')
print(kmeans.inertia_)
