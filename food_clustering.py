import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358S22/main/demo_25_Dimension_Reduction/protein.csv')

df.info()

df = df.drop('Country',axis=1)
df.isna().sum()

print(df.dtypes)
df.drop_duplicates(inplace=True)

df.describe()



#heatmap to show correlations amound the features in the dataset

import seaborn as sns

sns.heatmap(df.corr(), annot=True)
plt.show()


from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

df_scaled = ms.fit_transform(df)



from sklearn.decomposition import PCA


pca = PCA(n_components=2)

pca_scaled = pca.fit_transform(df_scaled)




from sklearn.cluster import KMeans


#appending iterations from 1-12 into a library titled 'cc' to to use the elbow method to find 
# optimal cluster

cc = []
for i in range(1,12):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=20,random_state=42).fit(pca_scaled)
    cc.append(kmeans.inertia_)


#The Elbow Method method to find optimal number of clusters


plt.plot(range(1,12),cc,marker='*')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method')
plt.show()


# kmeans using optimal point from elbow method
kmeans = KMeans(n_clusters=8,init='k-means++',n_init=20,random_state=42).fit(pca_scaled)
labels = kmeans.predict(pca_scaled)
df['Cluster'] = labels


# A scatterplot of k = 8
plt.scatter(pca_scaled[:,0],pca_scaled[:,1],c=df['Cluster'],marker='x',edgecolors='r')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='*',color='m')
plt.show()


# scoring
from sklearn.metrics import silhouette_score

sh = silhouette_score(pca_scaled,labels)
print(f'the sh score is: {sh*100:.2f}%')
print(kmeans.inertia_)




