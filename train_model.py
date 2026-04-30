import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import shutil
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models import infer_signature

def scale_frame(frame):
    df = frame.copy()
    X = df.drop(columns=['Sleep Disorder'])
    y = df['Sleep Disorder']
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    return X_scale, y, scaler

if __name__ == "__main__":
    base_path = "/var/lib/jenkins/workspace/Download"
    mlflow.set_tracking_uri(f"file://{base_path}/mlruns")
    
    df = pd.read_csv(f"{base_path}/df_clear.csv")
    X, y, scaler = scale_frame(df)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    mlflow.set_experiment("sleep_disorder_classification")

    params_sgd = {
        'alpha': [0.0001, 0.001, 0.01],
        'penalty': ['l2', 'l1'],
        'loss': ['log_loss', 'modified_huber']
    }
    with mlflow.start_run(run_name="SGD_Classifier"):
        sgd = SGDClassifier(random_state=42)
        clf_sgd = GridSearchCV(sgd, params_sgd, cv=3)
        clf_sgd.fit(X_train, y_train)
        best_sgd = clf_sgd.best_estimator_
        y_pred = best_sgd.predict(X_val)
        
        mlflow.log_params(clf_sgd.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_val, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_val, y_pred, average='weighted'))
        mlflow.sklearn.log_model(best_sgd, "model", signature=infer_signature(X_train, best_sgd.predict(X_train)))

    params_lr = {'C': [0.1, 1.0, 10.0]}
    with mlflow.start_run(run_name="Logistic_Regression"):
        lr = LogisticRegression(max_iter=1000, random_state=42)
        clf_lr = GridSearchCV(lr, params_lr, cv=3)
        clf_lr.fit(X_train, y_train)
        best_lr = clf_lr.best_estimator_
        y_pred = best_lr.predict(X_val)
        
        mlflow.log_params(clf_lr.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_val, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_val, y_pred, average='weighted'))
        mlflow.sklearn.log_model(best_lr, "model", signature=infer_signature(X_train, best_lr.predict(X_train)))

    params_rf = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}
    with mlflow.start_run(run_name="Random_Forest"):
        rf = RandomForestClassifier(random_state=42)
        clf_rf = GridSearchCV(rf, params_rf, cv=3)
        clf_rf.fit(X_train, y_train)
        best_rf = clf_rf.best_estimator_
        y_pred = best_rf.predict(X_val)
        
        mlflow.log_params(clf_rf.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_val, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_val, y_pred, average='weighted'))
        mlflow.sklearn.log_model(best_rf, "model", signature=infer_signature(X_train, best_rf.predict(X_train)))

    joblib.dump(scaler, f"{base_path}/scaler.pkl")

    current_experiment = mlflow.get_experiment_by_name("sleep_disorder_classification")
    exp_id = current_experiment.experiment_id
    
    dfruns = mlflow.search_runs(experiment_ids=[exp_id])
    best_run = dfruns.sort_values("metrics.accuracy", ascending=False).iloc[0]
    run_id = best_run.run_id
    
    output_dir = os.path.join(base_path, "best_model_dir")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    path2model = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=output_dir)
    
    with open(f"{base_path}/best_model.txt", "w") as f:
        f.write(path2model)
    
    print(path2model)
