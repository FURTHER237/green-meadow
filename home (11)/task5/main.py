import pandas as pd
import argparse
from lightgbm_model import train_lightgbm
from mlp_model import train_mlp

def main(model_type):
    # 选择不同的数据源
    if model_type == "lightgbm":
        df = pd.read_csv("../merged_cleaned.csv")
    elif model_type == "mlp":
        df = pd.read_csv("../merged_onehot.csv")

    else:
        raise ValueError("Unsupported model type. Use 'lightgbm' or 'mlp'.")

    # 提取特征和标签
    X = df.drop("SEVERITY", axis=1)
    y = df["SEVERITY"]

    # LightGBM 特殊处理类别特征
    if model_type == "lightgbm":
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype('category')

    # 选择对应训练函数
    model_func = {
        "lightgbm": train_lightgbm,
        "mlp": train_mlp
    }[model_type]

    # 模型训练与评估
    model, report, acc, f1 = model_func(X, y)
    print("Model:", model_type)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightgbm", help="Choose model: lightgbm or mlp")
    args = parser.parse_args()
    main(args.model)
