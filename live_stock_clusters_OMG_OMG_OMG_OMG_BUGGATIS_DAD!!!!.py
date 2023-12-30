import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf


stocks = ['AAPL','GOOGL','IBM','TGT','SBUX','TLSA','F','AMZN']

start_date = '2020-10-10'
end_date = '2023-12-29'


df_stocks = yf.download(tickers=stocks,start=start_date,end=end_date)

features = df_stocks[['Close','Adj Close','Volume','High']]

X = features

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X = ms.fit_transform(X)

from sklearn.cluster import KMeans

cc = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,random_state=0).fit(X)
    cc.append(kmeans.inertia_)



# Elbow method for best value of K

plt.plot(range(1,20),cc,marker='x')
plt.xlabel('Clusters')
plt.ylabel('inertia')
plt.title('Elbow Method')
plt.show()

k = 10


kmeans = KMeans(n_clusters=k,init='k-means++',n_init=10,random_state=0).fit(X)
labels = kmeans.labels_
df_stocks['clusters'] = labels


plt.scatter(X[:,0],X[:,1],c=df_stocks['clusters'],marker='*',edgecolors='r')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='k')
plt.xlabel('Closing Prices')
plt.ylabel('Trading Volume')
plt.show()


pred = kmeans.predict(X)

from sklearn.metrics import silhouette_score
sh = silhouette_score(X,pred)
print(f'score: {sh*100:.2f}%')
print(f'intertia: {kmeans.inertia_}')




