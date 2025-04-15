import pandas as pd
import json
import matplotlib.pyplot as plt

def task4_1():
    # read the csv data
    vehicle_df = pd.read_csv('../vehicle.csv')
    accident_df = pd.read_csv('../accident.csv')

    # extract year from accident date 
    accident_df['ACCIDENT_YEAR'] = pd.to_datetime(accident_df['ACCIDENT_DATE']).dt.year

    # merge the two dataset by 'ACCIDENT_NO'
    merged_df = pd.merge(vehicle_df, accident_df[['ACCIDENT_NO', 'ACCIDENT_YEAR']], on='ACCIDENT_NO', how='inner')

    # calculate the vehicle age by (manufature year - accident year)
    merged_df['VEHICLE_AGE'] = merged_df['ACCIDENT_YEAR'] - merged_df['VEHICLE_YEAR_MANUF']

    # group by vehicle age
    merged_df['VEHICLE_AGE_GROUP'] = merged_df['VEHICLE_AGE'].apply(lambda x: 'Old' if x > 10 else 'New')

    # count the number of vehicles in each group and output carstat.json
    age_group_counts = merged_df['VEHICLE_AGE_GROUP'].value_counts().to_dict()
    with open('task4_1_carstat.json', 'w') as f:
        json.dump(age_group_counts, f, indent=4)

    # group by accident year and age group to get yearly counts
    yearly_counts = merged_df.groupby(['ACCIDENT_YEAR', 'VEHICLE_AGE_GROUP']).size().unstack(fill_value=0)

    # out put stackbar.png
    plt.figure(figsize=(12, 6))
    yearly_counts.plot(kind='bar', stacked=True)

    plt.title('Accidents by Vehicle Age Group Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Accidents')
    plt.legend(title='Vehicle Age Group')
    plt.tight_layout()

    plt.savefig('task4_1_stackbar.png', dpi=300, bbox_inches='tight')
    plt.close()
