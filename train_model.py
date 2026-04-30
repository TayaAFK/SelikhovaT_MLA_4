import os
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

def scale_frame(frame):
    df = frame.copy()
    # Твой целевой признак — Sleep Disorder
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

    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'loss': ['log_loss', 'modified_huber']
    }

    mlflow.set_experiment("sleep_disorder_classification")

    with mlflow.start_run(run_name="SGD_Classifier"):
        model = SGDClassifier(random_state=42)
        clf = GridSearchCV(model, params, cv=3)
        clf.fit(X_train, y_train)
        
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        mlflow.log_params(clf.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
        # Сохраняем скалер
        joblib.dump(scaler, f"{base_path}/scaler.pkl")

    current_experiment = mlflow.get_experiment_by_name("sleep_disorder_classification")
    exp_id = current_experiment.experiment_id
    dfruns = mlflow.search_runs(experiment_ids=[exp_id])
    
    best_run = dfruns.sort_values("metrics.accuracy", ascending=False).iloc[0]
    run_id = best_run.run_id
    
    output_dir = os.path.join(base_path, "best_model_dir")
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir) # Очищаем старую модель
        
    path2model = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=output_dir)
    
    with open(f"{base_path}/best_model.txt", "w") as f:
        f.write(path2model)
    
    print(path2model)
