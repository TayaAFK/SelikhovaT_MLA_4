import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from mlflow.models import infer_signature

def scale_data(frame):
    df = frame.copy()
    # Предсказываем 'Sleep Disorder'
    X = df.drop(columns=['Sleep Disorder'])
    y = df['Sleep Disorder']
    
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    return X_scale, y, scaler

if __name__ == "__main__":
    df = pd.read_csv("./df_clear.csv")
    
    X, y, scaler = scale_data(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'loss': ['hinge', 'log_loss', 'modified_huber']
    }

    mlflow.set_experiment("sleep_disorder_classification")
    
    with mlflow.start_run():
        model = SGDClassifier(random_state=42)
        clf = GridSearchCV(model, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train)
        
        best_model = clf.best_estimator_
        y_pred = best_model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        prec = precision_score(y_val, y_pred, average='weighted')
        
        mlflow.log_params(clf.best_params_)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        joblib.dump(best_model, "sleep_model.pkl")
        joblib.dump(scaler, "scaler.pkl")

    dfruns = mlflow.search_runs()
    best_run = dfruns.sort_values("metrics.accuracy", ascending=False).iloc[0]
    path2model = best_run['artifact_uri'].replace("file://", "") + '/model'
    
    with open("best_model.txt", "w") as f:
        f.write(path2model)
    
    print(f"Best model path: {path2model}")
