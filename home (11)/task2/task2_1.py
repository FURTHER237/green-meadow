import pandas as pd

def task2_1():
    df = pd.read_csv("accident_dataset.csv")
    cleaned_descriptions = preprocess_text(df['DCA_DESC'])
    
    return
