# Imports
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Read in the train data
train_data = pd.read_csv('Data/train.csv')

# Read in the test data & prices and concat them
test_data  = pd.read_csv('Data/test.csv')
test_prices = pd.read_csv('Data/sample_submission.csv')
test_prices.drop(['Id'], axis = 1, inplace=True)
test_data = pd.concat([test_data, test_prices], axis=1)

# Merge the train and test, will be split later again.
data = pd.concat([train_data, test_data]).reset_index()

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

    dataframe  = data.copy()



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



# Apply the function
data_cleaned = clean_data(data, (100/3))


def one_hot_encoding(data):
    '''
    Function to convert data in dataframe to all numerical using onehot encoding.
    '''
    # Create a copy
    dataframe = data.copy()

    # Convert categorical data to numerical
    dataframe = pd.get_dummies(data=dataframe)

    # Return the new dataframe
    return dataframe



def label_encoding(data):
    '''
    Function to convert data in a dataframe to all numerical using label encoding
    '''

    # Create a copy
    dataframe = data.copy()

    # Go over all the features in the dataframe
    for feature in dataframe.columns:

        # If the data of the feature is categorical, apply label encoding for said column
        if is_string_dtype(dataframe[feature]):

            le = preprocessing.LabelEncoder()
            le.fit(dataframe[feature])
            dataframe[feature] = le.transform(dataframe[feature])

            # NOTE: Maybe make dictionary for saving the LabelEncoder classes, to convert the classes back to the original strings later.

    return dataframe


def split_x_y(data):
    '''
    Function to split data in x and y
    '''
    # Create a copy
    dataframe = data.copy()

    # Create y series
    y = dataframe['SalePrice'].reset_index(drop=True)

    # Drop the y from the x features
    dataframe.drop(['SalePrice'], axis = 1, inplace=True).reset_index(drop=True)

    return dataframe, y
