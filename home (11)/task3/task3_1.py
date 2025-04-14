import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def task3_1():
    # Read the filtered vehicle dataset
    df = pd.read_csv('../filtered_vehicle.csv')
    
    # Group by the three features and count crashes
    crash_counts = df.groupby(['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE']).size().reset_index(name='CRASH_COUNT')
    
    # Create scatterplot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=crash_counts, 
                   x='VEHICLE_YEAR_MANUF', 
                   y='CRASH_COUNT',
                   alpha=0.6)
    
    # Customize the plot
    plt.title('Vehicle Crashes by Year of Manufacture, Body Style, and Manufacturer')
    plt.xlabel('Year of Vehicle Manufacture')
    plt.ylabel('Number of Crashes')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('task3_1_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return
    
