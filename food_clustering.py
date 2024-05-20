import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358S22/main/demo_25_Dimension_Reduction/protein.csv')
df.info()
df = df.drop('Country',axis=1)


from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

df_scaled = ms.fit_transform(df)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_scaled = pca.fit_transform(df_scaled)


from sklearn.cluster import KMeans


cc = []
for i in range(1,12):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=20,random_state=1).fit(pca_scaled)
    cc.append(kmeans.inertia_)



import matplotlib.pyplot as plt

plt.plot(range(1,12),cc,marker='X')
plt.show()


# k = 9

kmeans = KMeans(n_clusters=9,init='k-means++',n_init=20,random_state=1).fit(pca_scaled)
labels = kmeans.labels_
df['Cluster 1'] = labels
df['Cluster 2'] = labels


plt.scatter(pca_scaled[:,0],pca_scaled[:,1],c=df['Cluster 1'],edgecolors='r',marker='X')
plt.scatter(pca_scaled[:,0],pca_scaled[:,1],c=df['Cluster 2'],edgecolors='m',marker='*')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='x',edgecolors='s')

plt.show()


from sklearn.metrics import silhouette_score

sh = silhouette_score(pca_scaled,labels)
print(f'the sh score is: {sh*100:.2f}%')
print('inertia: ',kmeans.inertia_)


