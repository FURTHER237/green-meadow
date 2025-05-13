import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def preprocess_data():
    # Load the datasets
    accident_df = pd.read_csv('../accident.csv')
    vehicle_df = pd.read_csv('../filtered_vehicle.csv')  # use filtered data is good enough
    person_df = pd.read_csv('../person.csv')
    
    # 1. Merge datasets using ACCIDENT_NO
    merged_df = pd.merge(accident_df, vehicle_df, on='ACCIDENT_NO', how='inner')
    merged_df = pd.merge(merged_df, person_df, on='ACCIDENT_NO', how='inner')
    
    # 2. Feature Engineering
    
    # Create severity index
    severity_map = {
        'Minor': 1,
        'Serious': 2,
        'Fatal': 3
    }
    merged_df['SEVERITY_INDEX'] = merged_df['SEVERITY'].map(severity_map)
    
    # Group continuous variables
    # Speed zones
    merged_df['SPEED_ZONE_CAT'] = pd.cut(
        merged_df['SPEED_ZONE'],
        bins=[0, 40, 60, 80, 100, float('inf')],
        labels=['Very Slow', 'Slow', 'Medium', 'Fast', 'Very Fast']
    )
    
    # Vehicle year
    current_year = pd.Timestamp.now().year
    merged_df['VEHICLE_YEAR'] = current_year - merged_df['VEHICLE_YEAR_MANUF']
    merged_df['VEHICLE_YEAR_CAT'] = pd.cut(
        merged_df['VEHICLE_YEAR'],
        bins=[1900, 2000, 2010, 2020, float('inf')],
        labels=['Very Old', 'Old', 'New', 'Very New']
    )

    # Person age_group - using the actual age groups from the data
    age_group_map = {
        '0-4': 'Child',
        '5-12': 'Child',
        '13-15': 'Teenager',
        '16-17': 'Teenager',
        '18-21': 'Young Adult',
        '22-25': 'Young Adult',
        '26-29': 'Adult',
        '30-39': 'Adult',
        '40-49': 'Middle-Aged',
        '50-59': 'Middle-Aged',
        '60-64': 'Senior',
        '65-69': 'Senior',
        '70+': 'Elderly',
        'Unknown': 'Unknown'
    }
    
    # Print age group distribution before mapping
    print("\nAge group distribution before mapping:")
    print(merged_df['AGE_GROUP'].value_counts())
    
    # Map age groups
    merged_df['PERSON_AGE_GROUP'] = merged_df['AGE_GROUP'].map(age_group_map)
    
    # Print age group distribution after mapping
    print("\nAge group distribution after mapping:")
    print(merged_df['PERSON_AGE_GROUP'].value_counts())
    
    # 3. Encoding categorical variables
    
    # Label Encoding for ordinal variables
    le = LabelEncoder()
    categorical_cols = [
        'ROAD_GEOMETRY_DESC',
        'LIGHT_CONDITION',
        'ACCIDENT_TYPE_DESC',
        'DAY_WEEK_DESC'
    ]
    
    for col in categorical_cols:
        merged_df[f'{col}_ENCODED'] = le.fit_transform(merged_df[col].fillna('Unknown'))
    
    # One-Hot Encoding for nominal variables
    nominal_cols = [
        'VEHICLE_BODY_STYLE',
        'VEHICLE_MAKE',
        'ROAD_USER_TYPE_DESC'
    ]
    
    for col in nominal_cols:
        dummies = pd.get_dummies(merged_df[col].fillna('Unknown'), prefix=col)
        merged_df = pd.concat([merged_df, dummies], axis=1)
    
    # 4. Outlier handling
    
    # Handle missing values in key fields
    # Age: Replace missing values with median
    if 'AGE' in merged_df.columns:
        merged_df['AGE'] = merged_df['AGE'].fillna(merged_df['AGE'].median())
    
    # Speed: Replace unrealistic values (e.g., > 200 km/h) with median
    speed_median = merged_df['SPEED_ZONE'].median()
    merged_df['SPEED_ZONE'] = merged_df['SPEED_ZONE'].apply(
        lambda x: speed_median if x > 200 or x < 0 else x
    )
    
    # Vehicle year: Replace missing values with median
    merged_df['VEHICLE_YEAR_MANUF'] = merged_df['VEHICLE_YEAR_MANUF'].fillna(
        merged_df['VEHICLE_YEAR_MANUF'].median()
    )
    
    # Drop original categorical columns that have been encoded
    columns_to_drop = categorical_cols + nominal_cols
    merged_df = merged_df.drop(columns=columns_to_drop)
    
    # Final validation
    print("\nFinal dataset validation:")
    print(f"Number of records: {len(merged_df)}")
    print(f"Number of features: {len(merged_df.columns)}")
    print("\nMissing values in key columns:")
    key_columns = ['SEVERITY_INDEX', 'SPEED_ZONE', 'VEHICLE_YEAR']
    if 'AGE' in merged_df.columns:
        key_columns.append('AGE')
    print(merged_df[key_columns].isnull().sum())
    
    return merged_df

if __name__ == "__main__":
    try:
        preprocessed_data = preprocess_data()
        print("\nData preprocessing completed successfully!")
        print(f"Final dataset shape: {preprocessed_data.shape}")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}") 