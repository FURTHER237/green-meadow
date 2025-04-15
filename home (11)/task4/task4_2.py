import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def task4_2():
    # read datasets
    vehicle_df = pd.read_csv('../vehicle.csv')
    accident_df = pd.read_csv('../accident.csv')

    # merge datasets on accident ID
    merged_df = pd.merge(vehicle_df[['ACCIDENT_NO', 'ROAD_SURFACE_TYPE_DESC']],
                         accident_df[['ACCIDENT_NO', 'ROAD_GEOMETRY_DESC']],
                         on='ACCIDENT_NO', how='inner')

    # count frequency of each combination
    crosstab = pd.crosstab(merged_df['ROAD_GEOMETRY_DESC'],
                           merged_df['ROAD_SURFACE_TYPE_DESC'])

    # plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd')

    plt.title('Accident Counts by Road Geometry and Surface Type')
    plt.xlabel('Road Surface Type')
    plt.ylabel('Road Geometry')
    plt.tight_layout()

    plt.savefig('task4_2_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    return