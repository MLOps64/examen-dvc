import pandas as pd
import logging
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Global varaiables
processed_path = "data/processed"
X_test_file = "X_test.csv"
X_train_file = "X_train.csv"
X_test_scaled_file = "X_test_scaled.csv"
X_train_scaled_file = "X_train_scaled.csv"



def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    #logging.info(f"=> processed file path :{processed_path}")
    #
    df_X_train = pd.read_csv(os.path.join(project_dir, processed_path, X_train_file), sep=',')
    df_X_test = pd.read_csv(os.path.join(project_dir, processed_path, X_test_file), sep=',')

    
    # Normalize
    X_train_scaled, X_test_scaled = normalize_data(df_X_train, df_X_test)
    # Save 
    save_dataframes(X_train_scaled, X_test_scaled,os.path.join(project_dir,processed_path))
    logging.info(f"=> Save data in {os.path.join(project_dir, processed_path)}")



def normalize_data(x_train, x_test):
    logging.info("=> Normalize data processus wait ...")

    # Standardisez les caractéristiques en supprimant la moyenne et en mettant à l’échelle la variance unitaire.
    scale = preprocessing.StandardScaler()

    #
    df_x_train = pd.DataFrame(scale.fit_transform(x_train), columns=x_train.columns)

    #
    df_x_test = pd.DataFrame(scale.fit_transform(x_test), columns=x_test.columns )

    #
    return df_x_train, df_x_test




def save_dataframes(X_train, X_test, output_folderpath):
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
    for file, filename in zip([X_train, X_test], [X_train_scaled_file, X_test_scaled_file]):
        output_filepath = os.path.join(output_folderpath, f'{filename}')
        # Write file
        file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    #
    project_dir = Path(__file__).resolve().parents[2]
    logging.info(f"=> Project dir : {project_dir}")
    #
    main(project_dir)