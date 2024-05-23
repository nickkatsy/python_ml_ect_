import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/Customer%20Data.csv')
df.isna().sum()
print(df.dtypes)

df.drop('CUST_ID',axis=1,inplace=True)
df.dtypes

df.dropna(inplace=True)


import seaborn as sns
import matplotlib.pyplot as plt


sns.heatmap(df.corr(), annot=True)
plt.show()



from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

df_scaled = ms.fit_transform(df)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_scaled = pca.fit_transform(df_scaled)


from sklearn.cluster import KMeans

cc = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=20,random_state=1).fit(pca_scaled)
    cc.append(kmeans.inertia_)


plt.plot(range(1,11),cc,marker='X')
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.show()


# k = 5 optimal number of clusters


kmeans = KMeans(n_clusters=5,init='k-means++',n_init=20,random_state=1).fit(pca_scaled)
labels = kmeans.labels_
df['Cluster 1'] = labels
df['Cluster 2'] = labels

plt.scatter(pca_scaled[:,0],pca_scaled[:,1],c=df['Cluster 1'],edgecolors='r',marker='X')
plt.scatter(pca_scaled[:,0],pca_scaled[:,1],c=df['Cluster 2'],edgecolors='m',marker='X')
plt.show()



from sklearn.metrics import silhouette_score

sh = silhouette_score(pca_scaled,labels)
print('sh score: ',sh*100)
print('inertia',kmeans.inertia_)





