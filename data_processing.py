# Imports
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



def check_missing(dataframe):
    '''
    Function to check what categories have missing data.
    '''

    # Check how many NaN values occur in a category
    nan_data  = dataframe.isna().sum().sort_values(ascending=False)

    # Make a dictionary out of the missing categories where the occurance of NaN > 0
    missing_categories = dict(nan_data.mask(nan_data == 0).dropna())

    # Return the dictionary
    return missing_categories



def clean_data(data, throwaway):
    '''
    Function to clean the data, decisions are based on how many NaN's a category has.

    Input: dataframe data, float throwaway (above what percentage the feature should be thrown away)
    '''

    # Copy the dataframe
    dataframe  = data.copy()

    # Get the categories with missing values
    missing_categories = check_missing(dataframe)
    
    # Drop the features with a lot of Nan data
    for feature in missing_categories:

        # The amount of NaN's in this category
        NaN_amount = missing_categories[feature]

        # If a category contains more NaN's than 1 third of the samples, remove the category
        if NaN_amount > (len(dataframe)/(1/throwaway*100)):
            dataframe.drop([feature], axis=1, inplace = True)

        # If it does not fall in the first condition, check if data is numerical or ordinal for the next step.
        elif is_string_dtype(dataframe[feature]):

            # For strings replace the NaN's with the mode value
            dataframe[feature].fillna(value=dataframe[feature].mode()[0], inplace=True)

        elif is_numeric_dtype(dataframe[feature]):

            # For numeric data replace the NaN's with the mean value
            dataframe[feature].fillna(value=dataframe[feature].mean(), inplace=True)


    # Check is all NaN  values have been filled
    assert len(check_missing(dataframe)) == 0, 'Still contains NaN'

    # Return the cleaned dataframe
    return dataframe



def one_hot_encoding(data):
    '''
    Function to convert data in dataframe to all numerical using onehot encoding.
    '''
    # Create a copy
    dataframe = data.copy()

    # Convert categorical data to numerical
    dataframe = pd.get_dummies(data=dataframe)

    # Return the edited dataframe
    return dataframe



def label_encoding(data):
    '''
    Function to convert data in a dataframe to all numerical using label encoding
    '''

    # Create a copy
    dataframe = data.copy()

    # Create a dictionary to save the LabelEncoder classes
    label_encoders = {}

    # Go over all the features in the dataframe
    for feature in dataframe.columns:

        # If the data of the feature is categorical, apply label encoding for said column
        if is_string_dtype(dataframe[feature]):

            le = preprocessing.LabelEncoder()
            le.fit(dataframe[feature])
            dataframe[feature] = le.transform(dataframe[feature])

            # NOTE: Maybe make dictionary for saving the LabelEncoder classes, to convert the classes back to the original strings later.

    # Return the edited dataframe
    return dataframe



def split_x_y(data):
    '''
    Function to split data in x and y
    '''
    # Create a copy
    dataframe = data.copy()

    # Create y series
    y = dataframe['SalePrice']

    # Drop the y from the x features
    dataframe.drop(['SalePrice'], axis = 1, inplace=True)

    # Return the dataframe without the price and the prices seperately
    return dataframe, y


def get_test_train(data_x, data_y):

    return train_test_split(data_x, data_y, test_size = 0.3, random_state=40)

def import_and_clean_train(file_name, encoder=one_hot_encoding):
    '''
    Imports, cleans and encodes the data. Returns x/y train/test sets.
    
    Inputs: file_name(str) and encoder(function, either 'one_hot_encoding' or 'label_encoding')
    Outputs: train_x, test_x, train_y, test_y
    '''

    # Create string
    location = 'Data/' + file_name + '.csv'

    # Read in the file
    df = pd.read_csv(location)
    
    # Clean the dataframe
    df_clean = clean_data(df)

    # Encode the ordinal data
    df_encoded = encoder(df_clean)

    # Create an x and an y
    df_x, df_y = split_x_y(df_encoded)

    # Return the x/y train/test
    return get_test_train(df_x, df_y)