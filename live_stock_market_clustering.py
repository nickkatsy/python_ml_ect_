import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


stocks = ['AAPL','GOOGL','TSLA','IBM','MCD']

start_date = '2020-10-10'
end_date = '2023-12-31'

df_stocks = yf.download(tickers=stocks,start=start_date,end=end_date)


X = df_stocks[['Close','Adj Close','High','Low']]


print(X.info())
print(X.isna().sum())


from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()

X_scaled = ms.fit_transform(X)

print(X_scaled)


from sklearn.cluster import KMeans


cc = []
for i in range(2,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,random_state=0).fit(X_scaled)
    cc.append(kmeans.inertia_)




import matplotlib.pyplot as plt

#elbow method for optimal number of clusters

plt.plot(range(2,11),cc,marker='o')
plt.xlabel('Cluster')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


#optimal numver of cluster is at 8

k = 8

kmeans = KMeans(n_clusters=k,init='k-means++',n_init=10,random_state=0).fit(X_scaled)
labels = kmeans.labels_
X['Cluster'] = labels

#visuals of clusters

plt.scatter(X_scaled[:,0],X_scaled[:,1],X_scaled[:,2],c=X['Cluster'],edgecolors='red',marker='x')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],marker='x',color='c')
plt.xlabel('Scaled Close Price')
plt.ylabel('Scaled Adjusted Price')
plt.show()

#scoring
pred = kmeans.predict(X_scaled)
from sklearn.metrics import silhouette_score
sh = silhouette_score(X_scaled,pred)
print(f'sh score: {sh*100:.2f}%')
print(kmeans.inertia_)




