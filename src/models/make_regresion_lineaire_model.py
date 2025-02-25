import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics as metrics
import joblib
import numpy as np
import logging
import os
from pathlib import Path

# Global
input_path = "data/processed"
output_path = "models"


def main(project_dir,input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logging.info(f"=> input file path :{input_filepath}, output file path : {output_filepath}")

    #
    LinearRegressionNotNormelize = 'LinearRegressionNotNormelize'
    LinearRegressionNormelize = 'LinearRegressionNormelize'
    #
    X_test, X_test_scaled, y_test, X_train, X_train_scaled, y_train = get_train_data(project_dir,input_filepath)
    
    #
    logging.info(f"=> X_train        : {X_train.shape}")
    logging.info(f"=> X_train_scaled : {X_train_scaled.shape}")
    logging.info(f"=> y_train        : {y_train.shape}")
    #
    logging.info(f"=> X_test         : {X_test.shape}")
    logging.info(f"=> X_test_scaled  : {X_test_scaled.shape}")
    logging.info(f"=> y_test         : {y_test.shape}")

    # Build model Not Optimized
    model_not_normelize = build_model(LinearRegressionNotNormelize,X_train,y_train,None)
    hyper_params = model_not_normelize.get_params()
    logging.info(f" Hypeparametre ; {hyper_params}")
    # 
    predict_not_opt, mse_not_opt, r2_score_not_opt = predict_model_scoring(LinearRegressionNotNormelize,model_not_normelize,X_test,y_test)
    #
    logging.info(f"=> The first five prediction with {LinearRegressionNotNormelize}: \n {predict_not_opt[:5]}")
    logging.info(f"=> The real five prediction with {LinearRegressionNotNormelize}: \n {y_test[:5]}")
    logging.info(f"=> Mean Squared Error (mse)  with {LinearRegressionNotNormelize} is  {mse_not_opt}")
    logging.info(f"=> r2_score  with {LinearRegressionNotNormelize} is  {r2_score_not_opt}")


    # Build model Optimized
    model_normelize = build_model(LinearRegressionNormelize,X_train_scaled,y_train,None)
    # 
    predict_opt, mse_opt, r2_score_opt = predict_model_scoring(LinearRegressionNormelize,model_normelize,X_test_scaled,y_test)
    #
    logging.info(f"=> The first five prediction with {LinearRegressionNormelize}: {predict_opt[:5]}")
    logging.info(f"=> The real five prediction with {LinearRegressionNormelize}: {y_test[:5]}")
    logging.info(f"=> Mean Squared Error (mse)  with {LinearRegressionNormelize} is  {mse_opt}")
    logging.info(f"=> r2_Scoree  with {LinearRegressionNormelize} is  {r2_score_opt}")
    
    # GridSeach
    params_space = {'copy_X': [True,False], 
               'fit_intercept': [True,False], 
               'n_jobs': [1,5,10,15,None], 
               'positive': [True,False]}
    
    #
    best_params,best_score = grid_search(model_normelize,params_space,X_train,y_train)
    logging.info(f"=> best_params : {best_params} - best_score :  {best_score}")

    #best param https://developers.google.com/machine-learning/crash-course/linear-regression/hyperparameters?hl=fr
    best_params = {'positive': False, 'n_jobs': None, 'fit_intercept': True, 'copy_X': True}
    model_best_params = build_model(LinearRegressionNormelize,X_train_scaled,y_train,best_params)

    #
    predict_opt, mse_opt, r2_score_opt = predict_model_scoring(LinearRegressionNormelize,model_best_params,X_test_scaled,y_test)
    #
    logging.info(f"=> The first five prediction with {LinearRegressionNormelize}: {predict_opt[:5]}")
    logging.info(f"=> The real five prediction with {LinearRegressionNormelize}: {y_test[:5]}")
    logging.info(f"=> Mean Squared Error (mse)  with {LinearRegressionNormelize} is  {mse_opt}")
    logging.info(f"=> r2_Scoree  with {LinearRegressionNormelize} is  {r2_score_opt}")

    #
    save_model(project_dir,output_filepath,"linear_not_normelized.joblib",model_not_normelize)
    save_model(project_dir,output_filepath,"linear_normelized.joblib",model_normelize)
    save_model(project_dir,output_filepath,"linear_normelized_best_param.joblib",model_normelize)

def grid_search(model,params_space,X_train, y_train):
    logging.info("=> GridSearch")
    #
    random_search = RandomizedSearchCV(model, params_space, n_iter=30, cv=5)
    random_search.fit(X_train, y_train)

    #
    return random_search.best_params_, random_search.best_score_


def get_train_data(project_dir,input_filepath):
    #
    X_test = pd.read_csv(os.path.join(project_dir, input_filepath,'X_test.csv' ), sep=',')
    X_test_scaled = pd.read_csv(os.path.join(project_dir, input_filepath,'X_test_scaled.csv' ), sep=',')
    y_test = pd.read_csv(os.path.join(project_dir, input_filepath,'y_test.csv' ), sep=',')

    #
    X_train = pd.read_csv(os.path.join(project_dir, input_filepath,'X_train.csv' ), sep=',')
    X_train_scaled = pd.read_csv(os.path.join(project_dir, input_filepath,'X_train_scaled.csv' ), sep=',')
    y_train = pd.read_csv(os.path.join(project_dir, input_filepath,'y_train.csv' ), sep=',')

    #
    return X_test, X_test_scaled, y_test, X_train, X_train_scaled, y_train


def build_model(model_name,X_train,y_train,best_params):
    #{'positive': False, 'n_jobs': 1, 'fit_intercept': True, 'copy_X': True}
    logging.info(f"=> Build Model {model_name}")
    #
    if best_params is not None:
        lr = LinearRegression(positive=best_params['positive'],n_jobs=best_params['n_jobs'],fit_intercept=best_params['fit_intercept'], copy_X=best_params['copy_X'] )
    else: 
        lr = LinearRegression()

    #
    lr.fit(X_train,y_train)
    #
    return lr

def predict_model_scoring(model_name,model,x_test,y_test):
    #
    logging.info(f"=> Pedict Model {model_name}")

    #
    predict_y = model.predict(x_test)

    #
    mse = metrics.mean_squared_error(y_test, predict_y)
    r2_score = (metrics.r2_score(y_test,predict_y) * 100)

    #
    return predict_y, mse, r2_score



def scoring_model(model_name,model,):
    logging.info(f"=> Scoring Model {model_name}")

def save_model(project_dir,output_filepath,model_filename,model):
    logging.info(f"=> Save Model {model_filename}")
    joblib.dump(model,os.path.join(project_dir, output_filepath,model_filename))

if __name__ == '__main__':
    #
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    #
    project_dir = Path(__file__).resolve().parents[2]
    logging.info(f"=> Projec dir : {project_dir}")
    #
    main(project_dir, input_path, output_path)