import pandas as pd
import warnings
warnings.filterwarnings('ignore')


food = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358S22/main/demo_25_Dimension_Reduction/protein.csv')

food.info()
food.describe()
food.drop('Country',axis=1,inplace=True)
df = food.dropna()

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df_scaled = sc.fit_transform(df)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

from sklearn.cluster import KMeans

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,random_state=0).fit(df_scaled)
    inertia.append(kmeans.inertia_)



import matplotlib.pyplot as plt

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

k = 7

kmeans = KMeans(n_clusters=k,init='k-means++',n_init=10,random_state=0).fit(df_scaled)
labels = kmeans.labels_
df['cluster'] = labels

plt.scatter(df_pca[:,0],df_pca[:,1],c=df['cluster'],cmap='viridis',edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='x',s=300,linewidth=10,color='r')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('feature 2')

print(kmeans.inertia_)

from sklearn.metrics import silhouette_score


print(f'Inertia: {kmeans.inertia_}')

sa = silhouette_score(df_pca, labels)
print(f'Silhouette score-- {sa*100}%')
