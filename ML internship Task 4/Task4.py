import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\Aditya Nikam\Documents\ML internship Task 4\iris.data.csv"
dataset = pd.read_csv(file_path, header=None)

# Extract features and target labels
X = dataset.iloc[:, 0:4].values  # Features
y = dataset.iloc[:, 4].values    # Target labels (not used in clustering)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Assume 3 clusters for the Iris dataset
clusters = kmeans.fit_predict(X)

# Apply PCA to reduce dimensions to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a scatter plot to visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("K-means Clustering with PCA Visualization", fontsize=14)
plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
