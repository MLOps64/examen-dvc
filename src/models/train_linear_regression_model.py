import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import sklearn.metrics as metrics
import joblib
import numpy as np
import logging
import os
from pathlib import Path

# Global
input_path = "data/processed"
output_path = "models"


def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logging.info(f"=> input file path :{input_path}, output file path : {output_path}")

    #
    LinearRegressionNotNormelize = 'LinearRegressionNotNormelize'
    LinearRegressionNormelize = 'LinearRegressionNormelize'
    #
    X_test, X_test_scaled, y_test, X_train, X_train_scaled, y_train = get_train_data(project_dir)
    
    #
    logging.info(f"=> X_train        : {X_train.shape}")
    logging.info(f"=> X_train_scaled : {X_train_scaled.shape}")
    logging.info(f"=> y_train        : {y_train.shape}")
    #
    logging.info(f"=> X_test         : {X_test.shape}")
    logging.info(f"=> X_test_scaled  : {X_test_scaled.shape}")
    logging.info(f"=> y_test         : {y_test.shape}")

    # Load estimator for best param
    lr_best_params = (joblib.load(os.path.join(project_dir, output_path,'lr_grid_search_estimator.pkl' ))).best_params_
    logging.info(f"=> Hypeparametres ; {lr_best_params}")

    # build lr model
    lr = LinearRegression(positive=lr_best_params['positive'],n_jobs=lr_best_params['n_jobs'],fit_intercept=lr_best_params['fit_intercept'], copy_X=lr_best_params['copy_X'] )

    # train lr model
    lr.fit(X_train_scaled, y_train)

    # save lr model
    joblib.dump(lr,os.path.join(project_dir, output_path,"lr_model.joblib"))
    logging.info(f"=> Save Model ; {os.path.join(project_dir, output_path,'lr_model.joblib')}")


def get_train_data(project_dir):
    #
    X_test = pd.read_csv(os.path.join(project_dir, input_path,'X_test.csv' ), sep=',')
    X_test_scaled = pd.read_csv(os.path.join(project_dir, input_path,'X_test_scaled.csv' ), sep=',')
    y_test = pd.read_csv(os.path.join(project_dir, input_path,'y_test.csv' ), sep=',')

    #
    X_train = pd.read_csv(os.path.join(project_dir, input_path,'X_train.csv' ), sep=',')
    X_train_scaled = pd.read_csv(os.path.join(project_dir, input_path,'X_train_scaled.csv' ), sep=',')
    y_train = pd.read_csv(os.path.join(project_dir, input_path,'y_train.csv' ), sep=',')

    #
    return X_test, X_test_scaled, y_test, X_train, X_train_scaled, y_train


if __name__ == '__main__':
    #
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    #
    project_dir = Path(__file__).resolve().parents[2]
    logging.info(f"=> Projec dir : {project_dir}")
    #
    main(project_dir)