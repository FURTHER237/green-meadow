import pandas as pd
from lightgbm_model import train_lightgbm

<<<<<<< HEAD
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
=======
# 示例数据加载（用户需替换为实际数据加载代码）
# df = pd.read_csv("your_data.csv")
# X = df.drop("SEVERITY", axis=1)
# y = df["SEVERITY"]

# 示例：假设数据已经准备好
# categorical_features = ['DCA_CODE', 'REG_STATE', 'VEHICLE_TYPE']

# model, report = train_lightgbm(X, y, categorical_features=categorical_features)
# print(report)
>>>>>>> parent of 15f038d (把merge后的数据转为csv了)
