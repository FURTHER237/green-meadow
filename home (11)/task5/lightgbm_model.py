
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def train_lightgbm(X, y, categorical_features=None, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = lgb.LGBMClassifier(objective='multiclass', num_class=len(set(y)))
    model.fit(X_train, y_train, categorical_feature=categorical_features)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return model, report
