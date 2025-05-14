import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



CATEGORICAL_FEATURES = [
    'SEX', 'AGE_GROUP', 'INJ_LEVEL', 'SEATING_POSITION', 'HELMET_BELT_WORN',
    'ROAD_USER_TYPE', 'LICENCE_STATE', 'TAKEN_HOSPITAL', 'EJECTED_CODE',
    'VEHICLE_DCA_CODE', 'INITIAL_DIRECTION', 'ROAD_SURFACE_TYPE',
    'REG_STATE', 'VEHICLE_TYPE', 'CONSTRUCTION_TYPE',
    'FUEL_TYPE', 'FINAL_DIRECTION', 'TRAILER_TYPE', 'VEHICLE_COLOUR_1',
    'VEHICLE_COLOUR_2', 'INITIAL_IMPACT', 'LEVEL_OF_DAMAGE', 'TOWED_AWAY_FLAG',
    'TRAFFIC_CONTROL', 'ACCIDENT_TYPE', 'DAY_OF_WEEK',
    'LIGHT_CONDITION', 'POLICE_ATTEND', 'ROAD_GEOMETRY', 'RMA'
]

def train_lightgbm(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = lgb.LGBMClassifier(objective='multiclass', num_class=len(set(y)), n_estimators=100)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='multi_logloss',
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[
            early_stopping(stopping_rounds=10),
            log_evaluation(period=10)
        ]
    )


    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", acc)
    print("F1 Score (macro):", f1)
    print("Classification Report:\n", report)


    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("confusion_matrix_lgbm.png", dpi=300, bbox_inches='tight')
    plt.show()


    results = model.evals_result_
    plt.figure(figsize=(8, 4))
    plt.plot(results['training']['multi_logloss'], label='Train Loss')
    plt.plot(results['valid_1']['multi_logloss'], label='Test Loss')
    plt.title('LightGBM Log Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.savefig("logloss_curve_lgbm.png", dpi=300, bbox_inches='tight')
    plt.show()


    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    importance_df.to_csv(("../importance_lgbm.csv"))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title("Top 20 Feature Importances (by Split Frequency)")
    plt.tight_layout()
    plt.savefig("feature_importance_lgbm.png", dpi=300)
    plt.show()

    return model, report, acc, f1
