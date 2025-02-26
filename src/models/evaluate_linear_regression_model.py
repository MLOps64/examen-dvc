import pandas as pd
import sklearn.metrics as metrics
import joblib
import numpy as np
import logging
import os
from pathlib import Path
import json

# Global
input_path = "data/processed"
output_path = "models"

def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logging.info(f"=> input file path :{input_path}, output file path : {output_path}")

    # Load data 
    X_test_scaled, y_test = get_train_data(project_dir)

    # Load Model 
    lr = joblib.load(os.path.join(project_dir, output_path,'lr_model.joblib' ))

    #précision du modèle
    precision = lr.score(X_test_scaled, y_test)
    logging.info(f"=> Precision : {precision}")

    # Make predictions
    predicted = lr.predict(X_test_scaled)
    logging.info(predicted)
    
    # Summarize the fit of the model
    mse = np.mean((predicted-y_test)**2)
    logging.info(f"=> lr.intercept_ : {lr.intercept_}")
    logging.info(f"=> lr.coef_      : {lr.coef_}")
    logging.info(f"=> mse           : {mse}")
    logging.info(f"=> precision     : {lr.intercept_}")

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_test, predicted)
    mean_absolute_error=metrics.mean_absolute_error(y_test, predicted) 
    mse=metrics.mean_squared_error(y_test, predicted) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_test, predicted)
    median_absolute_error=metrics.median_absolute_error(y_test, predicted)
    r2=metrics.r2_score(y_test, predicted)
    # buil dict of metrics
    metric_dict = {}
    metric_dict['explained_variance'] = round(explained_variance,4)
    metric_dict['mean_squared_log_error'] = round(mean_squared_log_error,4)
    metric_dict['median_absolute_error'] = round(median_absolute_error,4)
    metric_dict['r2'] =  round(r2,4)
    metric_dict['MAE'] = round(mean_absolute_error,4)
    metric_dict['MSE'] = round(mse,4)
    metric_dict['RMSE'] = round(np.sqrt(mse),4)
    #
    logging.info(f"=> metric {metric_dict}")

    # Save metrics in json file
    with open(os.path.join(project_dir, 'metrics','lr_metrics.json'), 'w') as fp:
        json.dump(metric_dict, fp, indent=4)



def get_train_data(project_dir):
    #
    #X_test = pd.read_csv(os.path.join(project_dir, input_path,'X_test.csv' ), sep=',')
    X_test_scaled = pd.read_csv(os.path.join(project_dir, input_path,'X_test_scaled.csv' ), sep=',')
    y_test = pd.read_csv(os.path.join(project_dir, input_path,'y_test.csv' ), sep=',')

    #
    #X_train = pd.read_csv(os.path.join(project_dir, input_path,'X_train.csv' ), sep=',')
    #X_train_scaled = pd.read_csv(os.path.join(project_dir, input_path,'X_train_scaled.csv' ), sep=',')
    #y_train = pd.read_csv(os.path.join(project_dir, input_path,'y_train.csv' ), sep=',')

    #
    return X_test_scaled, y_test


if __name__ == '__main__':
    #
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    #
    project_dir = Path(__file__).resolve().parents[2]
    logging.info(f"=> Projec dir : {project_dir}")
    #
    main(project_dir)