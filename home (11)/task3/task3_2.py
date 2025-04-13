import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def task3_2():
    # Step 1: Read the filtered vehicle dataset
    # This loads the data from the CSV file into a pandas DataFrame
    df = pd.read_csv('../filtered_vehicle.csv')
    
    # Step 2: Define the numerical features we want to analyze
    # These are the vehicle characteristics we'll use for clustering
    numerical_features = [
        'NO_OF_WHEELS',        # Number of wheels on the vehicle
        'NO_OF_CYLINDERS',     # Number of engine cylinders
        'SEATING_CAPACITY',    # How many people the vehicle can seat
        'TARE_WEIGHT',         # Weight of the vehicle when empty
        'TOTAL_NO_OCCUPANTS'   # Number of people in the vehicle during crash
    ]
    
    # Step 3: Group the data by vehicle characteristics
    # We group by year, body style, and manufacturer, then calculate the mean
    # of each numerical feature for each group
    grouped_data = df.groupby(['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE'])[numerical_features].mean().reset_index()
    
    # Step 4: Normalize the numerical features
    # This ensures all features have equal weight in the clustering
    # MinMaxScaler transforms features to a range between 0 and 1
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(grouped_data[numerical_features])
    
    # Step 5: Calculate SSE (Sum of Squared Errors) for different k values
    # We'll try k values from 1 to 10 to find the optimal number of clusters
    sse = []  # List to store SSE values
    k_range = range(1, 11)  # Try k values from 1 to 10
    
    for k in k_range:
        # Create a KMeans model with k clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        # Fit the model to our normalized data
        kmeans.fit(normalized_features)
        # Store the SSE (inertia_) for this k value
        sse.append(kmeans.inertia_)
    
    # Step 6: Create the elbow plot
    # This helps us visualize the optimal number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, 'bx-')  # Plot k vs SSE with blue x markers
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    
    # Step 7: Save the plot
    plt.savefig('task3_2_elbow.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return
