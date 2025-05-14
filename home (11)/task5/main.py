import pandas as pd
from lightgbm_model import train_lightgbm

def main():
    df = pd.read_csv("../merged_cleaned.csv")  
    X = df.drop("SEVERITY", axis=1)
    y = df["SEVERITY"]

    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')


    model, report, acc, f1 = train_lightgbm(X, y)
    print(report)

if __name__ == "__main__":
    main()
