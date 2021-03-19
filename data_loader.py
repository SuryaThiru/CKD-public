import os
import gzip

import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm
from pandas import read_feather
from tensorflow.keras import utils
from pickle import load


class DataLoader(utils.Sequence):
    """
    Class loads time series loads data from local disk into keras environment with relevant options
    
    """

    def __init__(
            self,
            list_ids,
            data_dir,
            month_dir,
            include_drug=True,
            batch_size=32,
            shuffle=True,
            by_month=True,
            preprocess_func=None):
        """
        data_dir : location of data, must contain drug and diag dirs, with patient info in root
        list_ids : list of IDs to load from disk
        include_drug : concat drug as features
        by_month :
        preprocess_func : function that takes arguments X, X_, y (features, age_sex, label)
                            in batches and returns the same after preprocessing
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.include_drug = include_drug
        self.by_month = by_month
        self.preprocess_func = preprocess_func

        self.data_dir = data_dir
        self.month_dir = month_dir

        self.list_ids = list_ids  # patients ids
        self.indexes = np.arange(len(self.list_ids))
        self.dataset = {'diag': []}  # list of csr matrices

        if self.include_drug:
            self.dataset['drug'] = []

        # load the dataset as sparse matrices
        self.__load_dataset_as_csr(self.list_ids)

        # get age/sex data with id as row index & labels for the IDs, <int,int> map
        self.info, self.labels = self.__load_info_labels(self.list_ids)

        self.dim = self.__get_dimension()

        self.on_epoch_end()

    def get_labels(self):
        """
        A convenient function to get all the labels of the data. Works with shuffle.
        """
        include = int(np.floor(len(self.list_ids) / self.batch_size)
                      ) * self.batch_size  # exclude last batch
        ids = [self.list_ids[i] for i in self.indexes[:include]]
        labels = [self.labels[i] for i in ids]
        return labels

    def __len__(self):
        """
        Denotes the no of batches per epoch
        """
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data. Note: this index doesn't refer to the patient ids (list_ids)
        """
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, X_, y = self.__data_generation(list_ids_temp, indexes)

        # preprocess
        if self.preprocess_func:
            X, X_, y = self.preprocess_func(X, X_, y)

        return [X, X_], y

    def __data_generation(self, ids, idx):
        """
        Generates data containing batch_size samples
        ids : list of patient ids to load from disk
        idx : indexes of the list_ids for the batch
        """
        # Initialization
        X = np.empty((self.batch_size, *self.__get_dimension(True)))
        X_ = np.empty((self.batch_size, 2))  # age, sex
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, [id, ix] in enumerate(zip(ids, idx)):
            # Store data
            X[i, ] = self.__load_data_matrix(ix)
            X_[i, ] = self.info.loc[id].values
            # Store class
            y[i] = self.labels[id]

        if self.by_month:
            X = self.__aggregate_week_to_month(X)

        return X, X_, y

    def __load_dataset_as_csr(self, ids):
        """
        load dataset into the memory dataset attribute).
        Populates the dataset dictionary with csr arrays in the order of ids
        (pass in the list_ids attrib to preserve the unshuffled order)
        ids: patients ids (list_ids)
        """
        with open(self.data_dir+'/'+self.month_dir+'/diag_dict.pkl', 'rb') as f:
            diag_dict = load(f)

        if self.include_drug:
            with open(self.data_dir+'/'+self.month_dir+'/drug_dict.pkl', 'rb') as f:
                drug_dict = load(f)

        for id in tqdm(ids, desc='Loading sparse dataset'):
            #  csr_diag = self.__load_diag_file(id)
            csr_diag = diag_dict[id]
            self.dataset['diag'].append(csr_diag)

            if self.include_drug:
                #  csr_drug = self.__load_drug_file(id)
                csr_drug = drug_dict[id]
                self.dataset['drug'].append(csr_drug)

    def __load_data_matrix(self, idx):
        """
        return dense numpy arrays from the dataset attribute
        idx : index for (but not) list_ids attribute, select from indexes attribute
        """
        data_mat = self.dataset['diag'][idx].toarray()

        if self.include_drug:
            drug_mat = self.dataset['drug'][idx].toarray()
            data_mat = np.hstack((data_mat, drug_mat))

        return data_mat

    def __load_info_labels(self, ids):
        """
        Load age, sex of IDs and the labels
        ids: list of patient ids
        """
        case_df = read_feather(self.data_dir + '/case_age_sex.feather')
        case_df['ckd'] = 1
        control_df = read_feather(self.data_dir + '/control_age_sex.feather')
        if 'year' in control_df.columns:
            control_df.drop('year', inplace=True, axis=1)
        control_df['ckd'] = 0

        info = case_df.append(control_df)  # merge case and control
        info['sex'] = np.where(info['sex'] == 'M', 1, 0)  # encode sex

        # uses patient as index (helps .loc)
        info.set_index('id', inplace=True)
        # ensure order
        #  info = info.loc[ids]

        labels = dict(zip(info.index.tolist(), info['ckd']))
        info.drop('ckd', axis=1, inplace=True)

        return info, labels

    def __get_dimension(self, orig=False):
        """
        method to obtain the dimension of the data (only the features and not (age, sex))
        orig : if True returns the dimensions of the data unaggregated across time
        """
        data_mat = self.__load_data_matrix(0)  # load a random(0th elem) data
        if self.by_month and not orig:
            reshaped = data_mat.reshape(1, *data_mat.shape)
            data_mat = self.__aggregate_week_to_month(
                reshaped)[0]  # add batch dim and remove

        X = data_mat

        # preprocess
        if self.preprocess_func and not orig:
            X = data_mat.reshape(1, *data_mat.shape)
            X_ = self.info.loc[278308].values.reshape(1, 2)
            y = self.labels[278308]
            X, X_, y = self.preprocess_func(X, X_, y)
            return X.shape[1:]

        return X.shape

    def __aggregate_week_to_month(self, arr):
        """
        aggregate 3d np array of dimensions (batch, weeks, features) to (batch, months, features)
        aggregation directly sums by weeks of 4
        """
        if arr.ndim != 3:
            raise ValueError(
                f'Expected input dimensions is 3 got dimension {arr.ndim}')

        weeks = arr.shape[1]
        rounded_weeks = weeks - (weeks % 4)

        # clip off the extra weeks to round of to multiples of 4
        arr = arr[:, -rounded_weeks:, :]

        # sum by 4 weeks
        splits = np.array_split(arr, rounded_weeks // 4, axis=1)
        sums = np.hstack([np.sum(split, axis=1, keepdims=True)
                          for split in splits])

        return sums

    def on_epoch_end(self):
        #         self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)


""" utility functions """


def filenames_to_ids(path):
    """
    get IDs of all files in the path, note: ids are present in filenames
    """
    files = os.listdir(path)
    ids = [int(id.split('_')[0]) for id in files]
    return sorted(ids)


if __name__ == "__main__":
    # run script as main file to test module
    data_dir = "data/data_oct/frames"
    diag_ids = filenames_to_ids(data_dir + '/diag_npz')
    drug_ids = filenames_to_ids(data_dir + '/drug_npz')

    list_ids = sorted(list(set(diag_ids) & set(drug_ids)))

    # create data loader with necessary variables
    dl = DataLoader(list_ids, data_dir, batch_size=128, include_drug=False)

    from time import time

    st = time()

    for i in range(3):
        X, y = dl[i]
        X, X_ = X

    et = time()

    print('Time taken: ', (et - st))
    print(X.shape, X_.shape, y.shape)

