import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def task3_3():
    # Step 1: Read the filtered vehicle dataset
    # This loads the data from the CSV file into a pandas DataFrame
    df = pd.read_csv('../filtered_vehicle.csv')
    
    # Step 2: Define the numerical features for clustering
    # These are the same features we used in Task 3.2
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
    
    # Step 4: Count crashes for each group
    # This creates a separate DataFrame with crash counts for each vehicle group
    crash_counts = df.groupby(['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE']).size().reset_index(name='CRASH_COUNT')
    
    # Step 5: Merge the grouped data with crash counts
    # This combines the numerical features with their corresponding crash counts
    final_data = pd.merge(grouped_data, crash_counts, 
                         on=['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE'])
    
    # Step 6: Normalize the numerical features
    # This ensures all features have equal weight in the clustering
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(final_data[numerical_features])
    
    # Step 7: Perform K-means clustering
    # We use k=3 clusters (based on the elbow plot from Task 3.2)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(normalized_features)
    
    # Step 8: Add cluster labels to the data
    # This allows us to color the points by cluster in the scatterplot
    final_data['CLUSTER'] = clusters
    
    # Step 9: Create colored scatterplot
    # This is similar to Task 3.1 but with colors representing clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=final_data, 
                   x='VEHICLE_YEAR_MANUF', 
                   y='CRASH_COUNT',
                   hue='CLUSTER',  # Color points by cluster
                   palette='viridis',  # Use a colorblind-friendly palette
                   alpha=0.6)  # Make points slightly transparent
    
    # Step 10: Customize the plot
    plt.title('Vehicle Crashes by Year of Manufacture (Colored by Cluster)')
    plt.xlabel('Year of Vehicle Manufacture')
    plt.ylabel('Number of Crashes')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Cluster')  # Add a legend showing cluster colors
    
    # Step 11: Save the plot
    plt.savefig('task3_3_scattercolour.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Step 12: Output top 10 vehicles for each cluster
    # For each cluster, find the 10 vehicles with highest crash counts
    for cluster_num in range(3):  # We have 3 clusters
        # Get data for this cluster
        cluster_data = final_data[final_data['CLUSTER'] == cluster_num]
        # Sort by crash count and take top 10
        top_10 = cluster_data.nlargest(10, 'CRASH_COUNT')
        
        # Define columns to include in the output
        output_cols = ['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE', 
                      'CRASH_COUNT'] + numerical_features
        
        # Save to CSV
        top_10[output_cols].to_csv(f'task3_3_cluster{cluster_num}.csv', index=False)
    
    return
