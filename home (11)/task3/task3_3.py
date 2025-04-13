import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def task3_3():
    df = pd.read_csv('../filtered_vehicle.csv')
    
    numerical_features = [
        'NO_OF_WHEELS',        
        'NO_OF_CYLINDERS',     
        'SEATING_CAPACITY',    
        'TARE_WEIGHT',         
        'TOTAL_NO_OCCUPANTS'  
    ]
    
    # We group by year, body style, and manufacturer, then calculate the mean
    # of each numerical feature for each group
    grouped_data = df.groupby(['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE'])[numerical_features].mean().reset_index()
    
    # Count crashes for each group
    crash_counts = df.groupby(['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE']).size().reset_index(name='CRASH_COUNT')
    
    # Merge the grouped data with crash counts
    final_data = pd.merge(grouped_data, crash_counts, 
                         on=['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE'])
    
    # Normalize the numerical features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(final_data[numerical_features])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(normalized_features)
    
    # Add cluster labels to the data
    final_data['CLUSTER'] = clusters
    
    # Create colored scatterplot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=final_data, 
                   x='VEHICLE_YEAR_MANUF', 
                   y='CRASH_COUNT',
                   hue='CLUSTER', 
                   palette='viridis',  
                   alpha=0.6)  
    
    # Customize the plot
    plt.title('Vehicle Crashes by Year of Manufacture (Colored by Cluster)')
    plt.xlabel('Year of Vehicle Manufacture')
    plt.ylabel('Number of Crashes')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Cluster')  
    
    # Save the plot
    plt.savefig('task3_3_scattercolour.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Output top 10 vehicles for each cluster
    for cluster_num in range(3):  
        cluster_data = final_data[final_data['CLUSTER'] == cluster_num]
            
        top_10 = cluster_data.nlargest(10, 'CRASH_COUNT')
        
        output_cols = ['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE', 
                      'CRASH_COUNT'] + numerical_features
        
        
        top_10[output_cols].to_csv(f'task3_3_cluster{cluster_num}.csv', index=False)
    
    return
