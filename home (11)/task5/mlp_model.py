import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import numpy as np
def train_mlp(X, y):
    from sklearn.inspection import permutation_importance
    import pandas as pd

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=True
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    plt.figure(figsize=(8, 4))
    plt.plot(model.loss_curve_, label='Training Loss')
    plt.title('MLP Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("logloss_curve_mlp.png", dpi=300, bbox_inches='tight')
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig("confusion_matrix_mlp.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Calculating permutation importance (n_repeats=3)...")
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=3,
        n_jobs=-1,
        scoring='accuracy'
    )
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance_mean', ascending=False)

    importance_df.to_csv("feature_importance_mlp_permutation.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(20), x='importance_mean', y='feature', xerr=importance_df.head(20)['importance_std'])
    plt.title("Top 20 Feature Importances (Permutation Importance)")
    plt.tight_layout()
    plt.savefig("feature_importance_mlp_permutation.png", dpi=300)
    plt.show()

    return model, report, acc, f1
