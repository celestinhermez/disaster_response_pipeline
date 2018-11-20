import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    A function to load our two datasets, returning a merged dataframe

    :param messages_filepath: a string with the filepath to the file with the messages. This file has to be a CSV,
    and to be in the same folder as our script
    :param categories_filepath: a string with the filepath to the file with the categories of each message.
    This file has to be a CSV, and to be in the same folder as our script

    :return: tw
    '''

    messages = pd.read_csv('messages.csv')
    categories = pd.read_csv('categories.csv')

    df = messages.merge(categories, on ='id', how='inner')

    return df

def clean_data(df):
    '''
    A function to clean our merged dataframe.

    :param df: a merged dataframe of messages and the associated categories
    :return: the same dataframe, cleaned
    '''

    # We isolate the categories from the categories column
    categories = df.categories.str.split(';', expand=True)

    # We use the first row for column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # We only keep the 0 and 1 values
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # We replace the categories column from our original dataframe with these new columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # We finish our cleaning by removing duplicates
    # and rows where the "related" category is 2 (values should only be 0 and 1

    df.drop_duplicates(inplace=True)
    df = df.loc[df.related != 2,:]

    return(df)

def save_data(df, database_filename):
    '''
    Stores the dataframe to a SQLLite database

    :param df: the dataframe which we want to save to a SQLLite database
    :param database_filename: the name of the database where we want to store our dataframe
    :return: None
    '''

    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql = ('messages_categorized', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()