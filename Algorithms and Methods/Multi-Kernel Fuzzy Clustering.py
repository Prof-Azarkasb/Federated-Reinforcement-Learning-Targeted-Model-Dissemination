import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load and Preprocess Dataset
# -------------------------------
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Select a few numerical features for clustering (adjust based on dataset)
    features = ['priority', 'average_usage', 'maximum_usage', 'assigned_memory', 'cpu_usage_distribution']
    df = df[features].dropna()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)

    return scaled_features

# -------------------------------
# Step 2: Compute Multi-Kernel Similarity
# -------------------------------
def compute_multi_kernel_matrix(X, gamma=0.5, degree=3):
    K1 = rbf_kernel(X, gamma=gamma)
    K2 = linear_kernel(X)
    K3 = polynomial_kernel(X, degree=degree)

    # Combine kernels with equal weights (can be tuned)
    return (K1 + K2 + K3) / 3

# -------------------------------
# Step 3: Multi-Kernel Fuzzy Clustering (MKFC)
# -------------------------------
def mkfc_clustering(X, n_clusters=3, max_iter=100, m=2.0, epsilon=1e-5):
    N = X.shape[0]
    K = compute_multi_kernel_matrix(X)

    # Initialize membership matrix U randomly
    U = np.random.dirichlet(np.ones(n_clusters), size=N)
    iteration = 0

    while iteration < max_iter:
        U_old = U.copy()
        # Compute cluster centers in kernel space
        centers = np.zeros((n_clusters, N))
        for k in range(n_clusters):
            numerator = np.sum((U[:, k][:, None] ** m) * K, axis=0)
            denominator = np.sum(U[:, k] ** m)
            centers[k] = numerator / denominator

        # Compute distances to cluster centers
        D = np.zeros((N, n_clusters))
        for i in range(N):
            for k in range(n_clusters):
                D[i, k] = np.linalg.norm(K[i] - centers[k])

        # Update U matrix
        for i in range(N):
            for k in range(n_clusters):
                denom = np.sum([(D[i, k] / D[i, j]) ** (2 / (m - 1)) for j in range(n_clusters)])
                U[i, k] = 1.0 / denom

        # Check for convergence
        if np.linalg.norm(U - U_old) < epsilon:
            break
        iteration += 1

    labels = np.argmax(U, axis=1)
    return labels, U

# -------------------------------
# Step 4: Visualize (Optional for 2D)
# -------------------------------
def visualize_clusters(X, labels):
    X_pca = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30)
    plt.title("MKFC Cluster Visualization (PCA-reduced)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()

# -------------------------------
# Step 5: Run Full Pipeline
# -------------------------------
if __name__ == "__main__":
    # Replace this with your actual Kaggle dataset path
    filepath = "google-2019.csv"
    
    X = load_and_prepare_data(filepath)
    labels, membership = mkfc_clustering(X, n_clusters=3)
    visualize_clusters(X, labels)
