import pandas as pd
import logging
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

input_path = "data/raw"
output_path = "data/processed"
cvs_file = "raw.csv"



def main(project_dir,input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logging.info(f"=> input file path :{input_filepath}, output file path : {output_filepath}")
    #
    df_raw = pd.read_csv(os.path.join(project_dir, input_path, cvs_file), sep=',')
    logging.info(f"=> Read {cvs_file}")
    # Normalize data

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = normalize_data(df_raw)
    # Save 
    save_dataframes(X_train, X_test,os.path.join(project_dir,output_path))
    logging.info(f"=> Save data in {os.path.join(project_dir, output_path)}")

def normalize_data(df):
    logging.info("=> Normalize data processus wait ...")
    # convert the datetime column to a pandas datetime object
    df['datetime'] = pd.to_datetime(df['date'], format="%Y-%m-%d  %H:%M:%S")

    # convert the datetime column to an integer
    df['timestamp'] = df['datetime'].astype(int)

    # divide the resulting integer by the number of nanoseconds in a second
    df['timestamp'] = df['timestamp'].div(10**9)
    
    # drop datime and date columns
    df.drop(columns=['date','datetime'], inplace=True) #inplace=True permet de mettre à jour df sans le re-affecter

    # reorganisation columns
    #df = df.reindex(columns=[])

    #
    logging.info(f"=> Transforme Date to timestamp : {df.info()}")
    df_features = df.drop(['silica_concentrate'], axis=1)
    df_target = df['silica_concentrate']
    # 
    transfomer = preprocessing.StandardScaler().fit(df_features)
    # slipt
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.2)
    return X_train, X_test, y_train, y_test




def save_dataframes(X_train, X_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        logging.info(f"=> {output_filepath}")
        #if not os.path.isfile(output_filepath):
        file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    #
    project_dir = Path(__file__).resolve().parents[2]
    logging.info(f"=> Projec dir : {project_dir}")
    #
    main(project_dir, input_path, output_path)