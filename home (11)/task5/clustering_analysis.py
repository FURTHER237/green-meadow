import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
        
        # Handle missing values separately for numerical and categorical columns
        for col in X.columns:
            if col in categorical_cols:
                # For categorical columns, fill with mode
                X[col] = X[col].fillna(X[col].mode().iloc[0])
            else:
                # For numerical columns, fill with mean
                X[col] = X[col].fillna(X[col].mean())
        
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
    
    def analyze_time_location_patterns(self):
        """Analyze time and location-based patterns."""
        # Convert time features
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Hour'] = self.data['Date'].dt.hour
        self.data['Day_of_Week'] = self.data['Date'].dt.dayofweek
        
        features = ['Hour', 'Day_of_Week', 'Latitude', 'Longitude']
        X_scaled, X = self.preprocess_for_clustering(features)
        
        # Perform DBSCAN clustering
        clusters, dbscan = self.perform_dbscan(X_scaled, eps=0.3, min_samples=5)
        
        # Create heatmap of accident hotspots
        self.create_accident_heatmap(clusters)
        
        # Analyze temporal patterns
        X['Cluster'] = clusters
        temporal_analysis = X.groupby('Cluster').agg({
            'Hour': 'mean',
            'Day_of_Week': 'mean'
        }).round(2)
        print("\nTemporal Pattern Analysis:")
        print(temporal_analysis)
    
    def create_accident_heatmap(self, clusters):
        """Create a heatmap of accident locations."""
        # Create a map centered on the mean coordinates
        m = folium.Map(location=[self.data['Latitude'].mean(), 
                               self.data['Longitude'].mean()],
                      zoom_start=10)
        
        # Add heatmap layer
        heat_data = [[row['Latitude'], row['Longitude']] 
                    for index, row in self.data.iterrows()]
        HeatMap(heat_data).add_to(m)
        
        # Save the map
        m.save('accident_hotspots.html')
    
    def analyze_vehicle_crash_patterns(self):
        """Analyze vehicle characteristics and crash patterns."""
        features = ['Engine_Capacity_(CC)', 'Age_of_Vehicle', 'Vehicle_Type']
        X_scaled, X = self.preprocess_for_clustering(features, 
                                                   categorical_cols=['Vehicle_Type'])
        
        # Perform K-means clustering
        clusters, kmeans = self.perform_kmeans(X_scaled, n_clusters=3)
        
        # Evaluate and visualize
        metrics = self.evaluate_clustering(X_scaled, clusters)
        print("\nVehicle Crash Pattern Clustering Metrics:")
        print(metrics)
        
        self.visualize_clusters(X_scaled, clusters, features, "Vehicle Crash Patterns")
        
        # Analyze cluster characteristics
        X['Cluster'] = clusters
        cluster_analysis = X.groupby('Cluster').agg({
            'Engine_Capacity_(CC)': 'mean',
            'Age_of_Vehicle': 'mean'
        }).round(2)
        print("\nVehicle Cluster Characteristics:")
        print(cluster_analysis)

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
        clustering.analyze_driver_risk_profiles()
        
        print("\nPerforming Time and Location Pattern Analysis...")
        clustering.analyze_time_location_patterns()
        
        print("\nPerforming Vehicle Crash Pattern Analysis...")
        clustering.analyze_vehicle_crash_patterns()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check the error message above for details.")

if __name__ == "__main__":
    main() 