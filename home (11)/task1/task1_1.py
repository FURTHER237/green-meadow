import pandas as pd

def task1_1():
    df = pd.read_csv('../person.csv')
    
    #input with most frequency number
    helmet_mode = df['HELMET_BELT_WORN'].mode()[0]
    df['HELMET_BELT_WORN'].fillna(helmet_mode, inplace=True) 
    print(df.loc[978,['HELMET_BELT_WORN']])
    
    df = pd.get_dummies(df, columns=['SEX', 'ROAD_USER_TYPE_DESC'])
    print(df.columns)
    
    def new_age_group(age):
        if isinstance(age, str) and ('-' in age or age == '70+'):
            try:
                #this is date, funny
                if age == '5-12':
                    return 'Unknown'
                first_age = int(age.split('-')[0])
                if first_age < 16:
                    return 'Under 16'
                elif first_age <= 25:
                    return '17-25'
                elif first_age <= 39:
                    return '26-39'
                elif first_age <= 64:
                    return '40-64'
                else:
                    return '65+'
            except:
                return 'Unknown'
        else:
            return 'Unknown'

    df['AGE_GROUP'] = df['AGE_GROUP'].apply(new_age_group)
    print(df.loc[df['AGE_GROUP'] == 'Unknown', ['AGE_GROUP']])
    print(df.loc[383,['AGE_GROUP']])
    age_group_counts = df['AGE_GROUP'].value_counts()
    print("AGE_GROUP Distribution:")
    print(age_group_counts)
    return df
    
