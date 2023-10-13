import pandas as pd



food = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358S22/main/demo_25_Dimension_Reduction/protein.csv', index_col=0)



food.info()

print(food.describe())


from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()
X_food = scaler.fit_transform(food)


from sklearn.cluster import KMeans


grpMeat = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X_food)


import matplotlib.pyplot as plt



# Scatterplot with read and white meat (cluster)

plt.figure(figsize=(8, 6))
plt.scatter(X_food[:, 0], X_food[:, 1], c=grpMeat.labels_, cmap='rainbow')
plt.xlabel('Red_Meat')
plt.ylabel('White Meat')
plt.title('K-Means Clustering (3 Clusters)')
plt.show()

# K-Means cluster with 7 clusters
grpProtein = KMeans(n_clusters=7, n_init=50, random_state=0).fit(X_food)

# 7 clusters of foood
plt.figure(figsize=(8, 6))
plt.scatter(X_food[:, 0], X_food[:, 1])
plt.xlabel('Red')
plt.ylabel('White')
plt.title('K-Means Clustering (7 Clusters)')
plt.show()

#clustering based on Red Meat and Fish (7 clusters)
plt.figure(figsize=(8, 6))
plt.scatter(X_food[:, 0], X_food[:, 2], c=grpProtein.labels_, cmap='rainbow')
plt.xlabel('Red Meat')
plt.ylabel('Fish')
plt.title('K-Means Clustering (7 Clusters)')
plt.show()