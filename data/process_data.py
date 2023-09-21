"""
DATA PREPROCESSING

Script Execution via:
--> python process_data.py messages.csv categories.csv disaster_response.db

:param csv-file with disaster messages
:param csv-file with disaster categories
:param file path of sqlite database
"""

import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm


def load_data(file_path_messages, file_path_categories):
    """
    Load csv-data and return merged data frame
    :param file_path_messages: path of messages.csv 
    :param file_path_categories: path of categories.csv
    :return: df: merged data frame of messages and categories 
    """""
    messages = pd.read_csv(file_path_messages)
    categories = pd.read_csv(file_path_categories)

    # merge messages and categories on 'id'
    df = pd.merge(left=messages, right=categories, on='id')
    return df


def clean_data(df):
    """"
    Clean data of input data frame
    :param df: merged data frame of messages and categories
    :return df_clean: cleaned data frame
    """
    # create a data frame 36 category columns
    categories = df['categories'].str.split(';', expand=True)
    # extract a category names
    category_names = [col for col in categories.loc[0].str[:-2]]
    # rename the columns of `categories`
    categories.columns = category_names

    for col in categories:
        # set each value as last character of the string
        categories.loc[:, col] = categories[col].str[-1]

    # convert data types to np.int8
    categories = categories.astype(np.int8)
    # add 'id' column
    categories.loc[:, 'id'] = df['id']

    # drop rows where 'related' == 2
    categories = categories[categories['related'] != 2]

    # drop the initial categories column from df
    df = df.drop(columns='categories')
    # merge dataframe with adjusted categories dataframe
    df_cleaned = pd.merge(left=df, right=categories, on='id')

    # get categories columns except for 'id'
    cols = [col for col in categories if col != 'id']
    # Drop rows without label, i.e. 0 in all category columns
    df_cleaned = df_cleaned[~df_cleaned[cols].isnull().all(axis=1)]
    # drop duplicates
    df_cleaned = df_cleaned.drop_duplicates()
    return df_cleaned


def save_data(df, file_path_db):
    """
    Save input data frame as sqlite data base in file_path_db
    :param df: cleaned data frame
    :param file_path_db: file name for data base
    """
    engine = create_engine(f'sqlite:///{file_path_db}')
    df.to_sql('disaster_response.db', engine, index=False, if_exists='replace')

    # engine = create_engine(f'sqlite:///{file_path_db}')
    # df.to_sql('clean_messages', engine, if_exists='replace', index=False)


def main():
    """
    This function serves as an implementation of an ETL pipeline, with the following steps

    1. Extract data from .csv files
    2. Perform data cleaning and pre-processing
    3. Load the processed data into SQLite database
    """
    print(sys.argv)
    if len(sys.argv) == 4:

        file_path_messages, file_path_categories, file_path_db = sys.argv[1:]

        # Wrap functions calls with tqdm to display progress bar
        with tqdm(total=100, desc="Loading data", unit='') as pbar:
            df = load_data(file_path_messages, file_path_categories)
            pbar.update(100)

        with tqdm(total=100, desc="Cleaning data") as pbar:
            df = clean_data(df)
            pbar.update(100)

        with tqdm(total=100, desc="Saving sqlite data base") as pbar:
            save_data(df, file_path_db)
            pbar.update(100)

        print('Saved cleaned data to data base!')

    else:
        print('Provide file paths of messages.csv and categories.csv like so:\n'
              'python process_data.py data/messages.csv data/categories.csv data/disaster_response.db')


if __name__ == '__main__':
    main()
