"""
Script contains functions to preprocess the dataset
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator):
    """
    Class prepares the dataframe for training
    1. Drops unneccesary columns
    2. Drops zero variance cols in training set on train and test set

    use the `fit` method to fit with the training set and `transform` to preprocess the dataframe
    """
    def __init__(self):
        self.include_cols = []

    def _common_preprocess(self, data):
        """
        Execute the common preprocessing steps in train and test data
        @param data: dataframe with targets ('ckd')
        @return data: preprocessed dataframe, with targets
        """

        data = data.drop('id', axis=1)        
        data = data.drop(['17', '488', 'B01AF', 'H01AB'], axis=1, errors='ignore')

        # drop age outliers
        idx = data[(data['age'] > 99)].index
        data = data.drop(idx)

        # drop rows with CKD
        idx = data[((data['585'] != 0) | (data['586'] != 0)) &
                    (data['ckd'] == 0)].index
        data = data.drop(idx)
        data = data.drop(['585', '586'], axis=1)

        return data

    def fit_transform(self, data):
        """
        Fit the preprocessor on the training data
        @param data: dataframe with targets ('ckd')
        @return input: preprocessed dataframe, input for model
        @return target: preprocessed series, target for model
        """
        data = self._common_preprocess(data)

        # drop cols with only zeroes
        data = data.loc[:, (data != 0).any(axis = 0)]
        # common preprocessed columns become new include cols for transformers
        self.include_cols.extend(data.columns)

        X = data.drop('ckd', axis = 1)
        y = data['ckd']

        return X, y

    def transform(self, data):
        """
        Preprocess data based on fit information
        @param data: dataframe with targets ('ckd')
        @return input: preprocessed dataframe, input for model
        @return target: preprocessed series, target for model
        """
        data = self._common_preprocess(data)
        data = data.loc[:, self.include_cols]
        X = data.drop('ckd', axis = 1)
        y = data['ckd']

        return X, y
