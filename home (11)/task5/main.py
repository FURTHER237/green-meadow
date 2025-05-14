import pandas as pd
import argparse
from lightgbm_model import train_lightgbm
from mlp_model import train_mlp

def main(model_type):
    if model_type == "lightgbm":
        df = pd.read_csv("../merged_cleaned.csv")
    elif model_type == "mlp":
        df = pd.read_csv("../merged_onehot.csv")

    else:
        raise ValueError("Unsupported model type. Use 'lightgbm' or 'mlp'.")

    X = df.drop("SEVERITY", axis=1)
    y = df["SEVERITY"]

    if model_type == "lightgbm":
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype('category')

    model_func = {
        "lightgbm": train_lightgbm,
        "mlp": train_mlp
    }[model_type]

    model, report, acc, f1 = model_func(X, y)
    print("Model:", model_type)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightgbm", help="Choose model: lightgbm or mlp")
    args = parser.parse_args()
    main(args.model)
