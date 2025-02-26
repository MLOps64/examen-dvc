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
# GridSeach for LinearRegression
    params_space = {'copy_X': [True,False], 
               'fit_intercept': [True,False], 
               'n_jobs': [1,5,10,15,20,25,30,35,None], 
               'positive': [True,False]
    }

    # 
    X_train_scaled, y_train, X_train = get_train_data(project_dir)

    # LinearRegresion
    lr =  LinearRegression()

    # RandomizedSearchCV
    lr_randon_search = RandomizedSearchCV(estimator=lr,param_distributions=params_space, n_iter=30, cv=5, verbose=1)
    lr_randon_search.fit(X_train_scaled, y_train)
    #print(lr_randon_search.best_estimator_)
    logging.info(f"=> RandomizedSearchCV with Scaled     : best estimator : {lr_randon_search.best_estimator_} - best params : {lr_randon_search.best_params_} - best score : {lr_randon_search.best_score_}")
    #
    lr_randon_search.fit(X_train, y_train)
    #print(lr_randon_search.best_estimator_)
    logging.info(f"=> RandomizedSearchCV with Not Scaled : best estimator : {lr_randon_search.best_estimator_} - best params : {lr_randon_search.best_params_} - best score : {lr_randon_search.best_score_}")

    # GridSearchCV
    lr_grid_search = GridSearchCV(estimator=lr,param_grid=params_space, n_jobs=5, cv=5, verbose=1)
    lr_grid_search.fit(X_train_scaled, y_train)
    logging.info(f"=> GridSearchCV  with Scaled         :best estimator : {lr_grid_search.best_estimator_} - best param : {lr_grid_search.best_params_} - best score: {lr_grid_search.best_score_}")
    #
    lr_grid_search = GridSearchCV(estimator=lr,param_grid=params_space, n_jobs=5, cv=5, verbose=1)
    lr_grid_search.fit(X_train_scaled, y_train)
    logging.info(f"=> GridSearchCV  with Not Scaled     :best estimator : {lr_grid_search.best_estimator_} - best param : {lr_grid_search.best_params_} - best score: {lr_grid_search.best_score_}")


    # Save estimator
    save_estimator(project_dir,"lr_randon_search_estimator.pkl", lr_randon_search)
    save_estimator(project_dir,"lr_grid_search_estimator.pkl", lr_grid_search)

def get_train_data(project_dir):
    #
    #X_test = pd.read_csv(os.path.join(project_dir, input_path,'X_test.csv' ), sep=',')
    #X_test_scaled = pd.read_csv(os.path.join(project_dir, input_path,'X_test_scaled.csv' ), sep=',')
    #y_test = pd.read_csv(os.path.join(project_dir, input_path,'y_test.csv' ), sep=',')

    #
    X_train = pd.read_csv(os.path.join(project_dir, input_path,'X_train.csv' ), sep=',')
    X_train_scaled = pd.read_csv(os.path.join(project_dir, input_path,'X_train_scaled.csv' ), sep=',')
    y_train = pd.read_csv(os.path.join(project_dir, input_path,'y_train.csv' ), sep=',')

    #
    return X_train_scaled, y_train, X_train


def save_estimator(project_dir,model_filename,model):
    logging.info(f"=> Save Model {model_filename}")
    joblib.dump(model,os.path.join(project_dir, output_path,model_filename))


if __name__ == '__main__':
    #
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    #
    project_dir = Path(__file__).resolve().parents[2]
    logging.info(f"=> Projec dir : {project_dir}")

    #
    main(project_dir)