import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')

class AccidentClustering:
    def __init__(self, data_path):
        """Initialize the clustering analysis with data loading and preprocessing."""
        self.data = pd.read_csv(data_path)
        print("\nAvailable columns in the dataset:")
        print(self.data.columns.tolist())
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def convert_age_range_to_numeric(self, age_range):
        """Convert age range to numeric value (midpoint of range)."""
        try:
            if pd.isna(age_range):
                return None
                
            if isinstance(age_range, (int, float)):
                return float(age_range)
                
            if isinstance(age_range, str):
                # Handle ranges like "25-34"
                if '-' in age_range:
                    start, end = map(int, age_range.split('-'))
                    return (start + end) / 2
                # Handle "Under X" cases
                elif age_range.lower().startswith('under'):
                    return 12.5
                # Handle "Over X" cases
                elif age_range.lower().startswith('over'):
                    return 85
                # Handle single numbers
                else:
                    try:
                        return float(age_range)
                    except ValueError:
                        print(f"Warning: Could not convert age value: {age_range}")
                        return None
            return None
            
        except Exception as e:
            print(f"Error converting age range: {age_range}, Error: {str(e)}")
            return None

    def preprocess_for_clustering(self, features, categorical_cols=None):
        """Preprocess data for clustering analysis."""
        # Verify features exist in dataset
        missing_features = [f for f in features if f not in self.data.columns]
        if missing_features:
            raise ValueError(f"Features not found in dataset: {missing_features}")
            
        # Select features
        X = self.data[features].copy()
        
        # Convert age ranges to numeric values if AGE is in features
        if 'AGE' in features:
            print("\nConverting AGE values...")
            X['AGE'] = X['AGE'].apply(self.convert_age_range_to_numeric)
            print(f"AGE conversion complete. Sample values:\n{X['AGE'].head()}")
            print(f"AGE value range: {X['AGE'].min()} to {X['AGE'].max()}")
        
        # Handle categorical variables
        if categorical_cols:
            for col in categorical_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
        
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, X
    
    def perform_kmeans(self, X_scaled, n_clusters=5, random_state=42):
        """Perform K-means clustering."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(X_scaled)
        return clusters, kmeans
    
    def perform_dbscan(self, X_scaled, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        return clusters, dbscan
    
    def evaluate_clustering(self, X_scaled, clusters):
        """Evaluate clustering results using multiple metrics."""
        if len(np.unique(clusters)) > 1:  # Only calculate if more than one cluster
            silhouette = silhouette_score(X_scaled, clusters)
            calinski = calinski_harabasz_score(X_scaled, clusters)
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski
            }
        return None
    
    def visualize_clusters(self, X, clusters, feature_names, title):
        """Visualize clusters using PCA for dimensionality reduction."""
        # Reduce dimensions to 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
        plt.title(f'Cluster Visualization - {title}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter, label='Cluster')
        plt.show()
        
        # Print cluster sizes
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()
        print(f"\nCluster sizes for {title}:")
        print(cluster_sizes)
    
    def analyze_driver_risk_profiles(self):
        """Analyze driver demographics and risk profiles."""
        features = ['AGE_GROUP', 'SEX', 'ROAD_USER_TYPE', 'SEVERITY']
        X_scaled, X = self.preprocess_for_clustering(features, categorical_cols=['SEX', 'ROAD_USER_TYPE'])
        
        # Perform K-means clustering
        clusters, kmeans = self.perform_kmeans(X_scaled, n_clusters=4)
        
        # Evaluate clustering
        metrics = self.evaluate_clustering(X_scaled, clusters)
        print("\nDriver Risk Profile Clustering Metrics:")
        print(metrics)
        
        # Visualize clusters
        self.visualize_clusters(X_scaled, clusters, features, "Driver Risk Profiles")
        
        # Analyze cluster characteristics
        X['Cluster'] = clusters
        cluster_analysis = X.groupby('Cluster').agg({
            'AGE_GROUP': 'mean',
            'SEVERITY': 'mean'
        }).round(2)
        print("\nCluster Characteristics:")
        print(cluster_analysis)
    
    def plot_clusters_matplotlib(self, X, clusters, feature_names):
        """Create a simple matplotlib plot of the clusters."""
        # Reduce dimensions to 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        
        # Add labels and title
        plt.title('Time and Location Pattern Clusters', fontsize=14)
        plt.xlabel('Time Pattern Component (Light & Day)', fontsize=12)
        plt.ylabel('Location Pattern Component (Speed & Road)', fontsize=12)
        
        # Add colorbar
        plt.colorbar(scatter, label='Cluster')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig('time_location_clusters.png')
        plt.close()
        
        # Print feature importance
        print("\nFeature importance in PCA:")
        for i, feature in enumerate(feature_names):
            print(f"{feature}: {abs(pca.components_[0][i]):.3f}")

    def analyze_time_location_patterns(self):
        """Analyze time and location-based patterns."""
        print("analyzing time and location patterns")
        print("preprocessing data")
        
        features = ['LIGHT_CONDITION', 'DAY_OF_WEEK', 'SPEED_ZONE', 'ROAD_GEOMETRY']
        X_scaled, X = self.preprocess_for_clustering(features)
        
        # Perform DBSCAN clustering
        clusters, dbscan = self.perform_dbscan(X_scaled, eps=0.3, min_samples=5)
        
        # Create matplotlib plot
        self.plot_clusters_matplotlib(X_scaled, clusters, features)
        
        # Analyze temporal patterns
        X['Cluster'] = clusters
        temporal_analysis = X.groupby('Cluster').agg({
            'LIGHT_CONDITION': 'mean',
            'DAY_OF_WEEK': 'mean',
            'SPEED_ZONE': 'mean',
            'ROAD_GEOMETRY': 'mean'
        }).round(1)
        print("\nTemporal Pattern Analysis:")
        print(temporal_analysis)
    
    def create_accident_heatmap(self, clusters):
        """Create a heatmap of accident locations."""
        # Create a map centered on the mean coordinates
        m = folium.Map(location=[self.data['SPEED_ZONE'].mean(), 
                               self.data['ROAD_GEOMETRY'].mean()],
                      zoom_start=10)
        
        # Add heatmap layer
        heat_data = [[row['SPEED_ZONE'], row['ROAD_GEOMETRY']] 
                    for index, row in self.data.iterrows()]
        HeatMap(heat_data).add_to(m)
        
        # Save the map
        m.save('accident_hotspots.html')
    
    def plot_clusters(self, X, clusters, feature_names, title):
        """Plot clusters using PCA for dimensionality reduction."""
        # Reduce dimensions to 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.title(f'Cluster Visualization - {title}', fontsize=14)
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        plt.colorbar(scatter, label='Cluster')
        
        # Add cluster centers
        for cluster in np.unique(clusters):
            if cluster != -1:  # Skip noise points
                center = np.mean(X_pca[clusters == cluster], axis=0)
                plt.scatter(center[0], center[1], c='red', marker='x', s=200, linewidths=3)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Print cluster sizes
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()
        print(f"\nCluster sizes for {title}:")
        print(cluster_sizes)
        
        # Print feature importance
        print("\nFeature importance in PCA:")
        for i, feature in enumerate(feature_names):
            print(f"{feature}: {abs(pca.components_[0][i]):.3f}")


def main():
    try:
        # Initialize clustering analysis
        data_path = '../merged_cleaned.csv'
        print(f"Loading data from: {data_path}")
        clustering = AccidentClustering(data_path)
        
        # Wait for user to see the columns and confirm
        input("\nPress Enter to continue after reviewing the available columns...")
        
        # Perform different clustering analyses
        print("\nPerforming Driver Risk Profile Analysis...")
        #clustering.analyze_driver_risk_profiles()
        
        print("\nPerforming Time and Location Pattern Analysis...")
        clustering.analyze_time_location_patterns()
        
        
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check the error message above for details.")

if __name__ == "__main__":
    main() 