import pands as pd

def task4_1():
    return

import pandas as pd
import json
import matplotlib.pyplot as plt


def task4_1():
    vehicle_df = pd.read_csv('vehicle.csv')
    accident_df = pd.read_csv('accident.csv')

    merged_df = pd.merge(vehicle_df, accident_df, on='VEHICLE_ID')

    merged_df['VEHICLE_AGE'] = merged_df['ACCIDENT_YEAR'] - merged_df['VEHICLE_YEAR']

    merged_df['VEHICLE_AGE_GROUP'] = merged_df['VEHICLE_AGE'].apply(lambda x: 'Old' if x > 10 else 'New')

    age_group_counts = merged_df['VEHICLE_AGE_GROUP'].value_counts().to_dict()

    with open('task4_1_carstat.json', 'w') as json_file:
        json.dump(age_group_counts, json_file, indent=4)
    yearly_counts = merged_df.groupby(['ACCIDENT_YEAR', 'VEHICLE_AGE_GROUP']).size().unstack(fill_value=0)

    yearly_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Accidents by Vehicle Age Group Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Accidents')
    plt.legend(title='Vehicle Age Group')
    plt.tight_layout()

    plt.savefig('task4_1_stackbar.png')
    plt.close()
    return None
