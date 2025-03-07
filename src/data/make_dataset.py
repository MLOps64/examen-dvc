import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

input_path = "data/raw"
output_path = "data/processed"
cvs_file = "raw.csv"

def main(project_dir,input_filepath, output_filepath):
    #
    log = logging.getLogger("main")
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    log.info(f"=> input file path :{input_filepath}, output file path : {output_filepath}")
    #
    df_raw = pd.read_csv(os.path.join(project_dir, input_path, cvs_file), sep=',')
    log.info(f"=> Read {cvs_file}")
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df_raw)
    # Save 
    save_dataframes(X_train, X_test, y_train, y_test,os.path.join(project_dir,output_path))
    log.info(f"=> Save data in {os.path.join(project_dir, output_path)}")


def split_data(df):
    ## convert the datetime column to a pandas datetime object
    #df['datetime'] = pd.to_datetime(df['date'], format="%Y-%m-%d  %H:%M:%S")

    ## convert the datetime column to an integer
    #df['timestamp'] = df['datetime'].astype(int)

    ## divide the resulting integer by the number of nanoseconds in a second
    #df['timestamp'] = df['timestamp'].div(10**9)
    #
    ## drop datime, timestamp and date columns
    #df.drop(columns=['date','datetime','timestamp'], inplace=True) #inplace=True permet de mettre à jour df sans le re-affecter

    ## reorganisation columns
    ##df = df.reindex(columns=[])

    # Drop date column
    df.drop(columns=['date'], inplace=True) #inplace=True permet de mettre à jour df

    # Build df target silica_concentrate
    target = df['silica_concentrate']
    feats = df.drop(columns=['silica_concentrate'], axis=1)
    logging.info(f"=> features before split : {feats.head()}")
    #
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # check
    if os.path.exists(os.path.join(output_folderpath)) == False :  
        try:
            os.makedirs(os.path.join(output_folderpath))
            print(f"Nested directories '{os.path.join(output_folderpath)}' created successfully.")
        except FileExistsError:
            print(f"One or more directories in '{os.path.join(output_folderpath)}' already exist.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{os.path.join(output_folderpath)}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        logging.info(f"=> {output_filepath}")
        #if not os.path.isfile(output_filepath):
        file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    logging.info(f"=> Projec dir : {project_dir}")
    #
    main(project_dir, input_path, output_path)
    