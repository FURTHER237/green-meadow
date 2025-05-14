import pandas as pd
import argparse
from lightgbm_model import train_lightgbm
from mlp_model import train_mlp  # 假设你有一个新的 mlp_model.py 文件

def main(model_type):
    df = pd.read_csv("../merged_cleaned.csv")  
    X = df.drop("SEVERITY", axis=1)
    y = df["SEVERITY"]

    # 类别特征转换为 category 类型（LightGBM 支持，MLP 需要 one-hot 或 embedding）
    if model_type == "lightgbm":
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype('category')

    model_func = {
        "lightgbm": train_lightgbm,
        "mlp": train_mlp
    }.get(model_type)

    if model_func is None:
        raise ValueError("Unsupported model type. Use 'lightgbm' or 'mlp'.")

    model, report, acc, f1 = model_func(X, y)
    print("Model:", model_type)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightgbm", help="Choose model: lightgbm or mlp")
    args = parser.parse_args()
    main(args.model)
