# import libraries

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    This function loads data from csv files and merges to a single pandas dataframe
    
    Input:
    messages_filepath - filepath to messages csv file
    categories_filepath - filepath to categories csv file
    
    Returns:
    df - dataframe merging categories and messages
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how='outer', on=['id'])
    return df


def clean_data(df):
    '''
    clean_data
    This function split the categories column into separate columns
    
    Input:
    df - dataframe to be cleaned
    
    Returns:
    df - dataframe cleaned
    '''
    
    # create a dataframe of the 36 individual category columns
    categories =  df['categories'].str.split(';',n=-1,expand=True)
    # select the first row of the categories dataframe
    row = categories.head(1)
    #print(row)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    #category_colnames = row.replace(pat,repl)

    # start stop and step variables
    start, stop, step = 0, -2, 1
    category_colnames = row.apply(lambda x: x.str.slice(start, stop, step)).values[0]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
       
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    frames = [df, categories]
    df = pd.concat(frames,axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    #replace 2 value with 1 for related column
    df.loc[df['related'] > 1, 'related'] = 1
    
    return df


def save_data(df, database_filename):
    '''
    save_data
    This function saves the dataframe into a database file
    
    Input:
    df - dataframe to be saved
    database_filename - database file name that will contain the saved data
    
    Returns:
    none
    '''
   
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('data', engine, index=False, if_exists='replace')


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
              'messages.csv categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()