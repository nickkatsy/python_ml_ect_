import pandas as pd
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('C:/ml/python/data/Mall_Customers.csv',delimiter=',')

data.info()

data.isna().sum()


df = data.copy()
df.nunique()
df.describe()

df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
df.info()

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=True)
plt.show()


df.info()
df.columns

X = df[['Annual Income (k$)','Spending Score (1-100)']]



from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()

X = ms.fit_transform(X)

from sklearn.cluster import KMeans


#Elbow method to find optimal number of clusters

cs = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=20,random_state=42).fit(X)
    cs.append(kmeans.inertia_)
    


plt.plot(range(1,20),cs,marker='*')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# from elbow method, choose k=5 for optimal number of clusters


k = 5

kmeans = KMeans(n_clusters=k,init='k-means++',n_init=20,random_state=42).fit(X)
labels = kmeans.labels_
df['Cluster'] = labels


plt.scatter(X[:,0],X[:,1],c=df['Cluster'],marker='x',s=300,edgecolors='r')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='m')
plt.xlabel('Annual Income k$')
plt.ylabel('Spending Score 1-100')
plt.show()



#results

from sklearn.metrics import silhouette_score

pred = kmeans.predict(X)
Sil_Score = silhouette_score(X,pred)
print(f'Silhouette score: {Sil_Score*100:.2f}%')
print(f'Cluster Sum-of-squares: {kmeans.inertia_}')
print(kmeans.cluster_centers_)
