import pandas as pd

def task1_1():
    
    #I should say that I used gpt for finding bugs, and got .fillna(), get_dummies() and .unstacked().
    #while doing this assgnment, I become more and more famliar with the operations so I become more 
    #independent and go to read https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#
    #and https://matplotlib.org/stable/api/_tight_layout_api.html#module-matplotlib._tight_layout.
    #without ai, I may do one hot encoding change the original form to a big list, which will be much 
    #more complex. .fillna ()was redundant because I proved with code that only one value is empty.
    #unstacked is a good thing that simplified the visualization. It transformed data like tree to a table.
    #there is one thing I can't understand that happened in task1_2.py. when I wanted to try modify the
    #df_filtered['SEATING_POSITION'], there was warning like: modifying a slice of copy etc. I don't know
    #what to do and gpt very stupied, it just provided basic help. Finally I changed variable name and something
    #in my code and everthing goes ok.
    df = pd.read_csv('/course/person.csv')
    
    #input with most frequency number
    helmet_mode = df['HELMET_BELT_WORN'].mode()[0]
    df['HELMET_BELT_WORN'] = df['HELMET_BELT_WORN'].fillna(helmet_mode) 
    print(df.loc[978,['HELMET_BELT_WORN']])
    
    # to vector
    print(df.groupby('ROAD_USER_TYPE_DESC').size())
    df = pd.get_dummies(df, columns=['SEX', 'ROAD_USER_TYPE_DESC'])
    print(df.columns)
    print(df['ROAD_USER_TYPE_DESC_Drivers'])
    print("drivers' sum", df['ROAD_USER_TYPE_DESC_Drivers'].sum())
    # age group in to broader range
    age_before = df.groupby('AGE_GROUP').size()
    print("AGE_GROUP before broader:")
    print(age_before)
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
    age_group_counts = df.groupby('AGE_GROUP').size()
    print("AGE_GROUP Distribution:")
    print(age_group_counts)
    
    return df
