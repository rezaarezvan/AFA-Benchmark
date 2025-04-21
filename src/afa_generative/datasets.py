import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset, TensorDataset, Dataset
from sklearn import preprocessing
from sklearn.utils import shuffle


GITHUB_URL = 'https://raw.githubusercontent.com/iancovert/dynamic-selection/main/datasets/'
        
        
def data_split(dataset, val_portion=0.2, test_portion=0.2, random_state=0):
    '''
    Split dataset into train, val, test.
    
    Args:
      dataset: PyTorch dataset object.
      val_portion: percentage of samples for validation.
      test_portion: percentage of samples for testing.
      random_state: random seed.
    '''
    # Shuffle sample indices.
    rng = np.random.default_rng(random_state)
    inds = np.arange(len(dataset))
    rng.shuffle(inds)

    # Assign indices to splits.
    n_val = int(val_portion * len(dataset))
    n_test = int(test_portion * len(dataset))
    test_inds = inds[:n_test]
    val_inds = inds[n_test:(n_test + n_val)]
    train_inds = inds[(n_test + n_val):]

    # Create split datasets.
    test_dataset = Subset(dataset, test_inds)
    val_dataset = Subset(dataset, val_inds)
    train_dataset = Subset(dataset, train_inds)
    return train_dataset, val_dataset, test_dataset


def get_xy(dataset):
    '''
    Extract inputs (x) and outputs (y) from dataset object.
    
    Args:
      dataset: PyTorch dataset object.
    '''
    x, y = zip(*list(dataset))
    if isinstance(x[0], np.ndarray):
        return torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y))
    elif isinstance(x[0], torch.Tensor):
        if isinstance(y[0], (int, float)):
            return torch.stack(x), torch.tensor(y)
        else:
            return torch.stack(x), torch.stack(y)
    else:
        raise ValueError(f'not sure how to concatenate data type: {type(x[0])}')


def load_spam(features=None):
    # Load data.
    data_dir = os.path.join(GITHUB_URL, 'spam.csv')
    data = pd.read_csv(data_dir)
    
    # Set features.
    if features is None:
        features = np.array([f for f in data.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
            
    # Extract x, y.
    x = np.array(data.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(data['Outcome']).astype('int64')
    
    # Create dataset object.
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset


def load_diabetes(features=None):
    # Load data.
    data_dir = os.path.join(GITHUB_URL, 'diabetes.csv')
    data = pd.read_csv(data_dir)
    
    # Set features.
    if features is None:
        features = np.array([f for f in data.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
            
    # Extract x, y.
    x = np.array(data.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(data['Outcome']).astype('int64')
    
    # Create dataset object.
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset


def load_miniboone(features=None):
    # Load data.
    data_dir = os.path.join(GITHUB_URL, 'miniboone.csv')
    data = pd.read_csv(data_dir)
    
    # Set features.
    if features is None:
        features = np.array([f for f in data.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
            
    # Extract x, y.
    x = np.array(data.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(data['Outcome']).astype('int64')
    
    # Create dataset object.
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset


class Boston(object):
    def __init__(self, root_dir):
        '''
        This is to load the Boston data as a whole
        :param root_dir: The path where UCI data are stored
        :type root_dir: Str
        :param rs: The random seed for splitting the training/test data
        :type rs: Int
        '''
        self.path = os.path.join(root_dir, 'boston.csv')
        self.Data = pd.read_csv(self.path)
        self.Data_mat = self.Data.values
        self.obs_dim = self.Data_mat.shape[1]
        # Normalize the data between 1 and 2
        Data_std = preprocessing.scale(self.Data_mat)
        Data_std[Data_std == 0] = 0.01
        self.Data_mat = Data_std
        
        self.Data_mat=shuffle(self.Data_mat,random_state=42)


class base_UCI_Dataset(Dataset):
    '''
    Most simple dataset by explicit giving train and test data
    '''
    def __init__(self,data,transform=None,flag_GPU=True):
        self.Data=data
        self.flag_GPU=flag_GPU
        self.transform=transform
    def __len__(self):
        return self.Data.shape[0]
    def __getitem__(self, idx):
        sample=self.Data[idx,:]
        if self.transform and self.flag_GPU==True:
            sample=self.transform(sample)
            sample=sample.cuda()
        elif self.transform and not self.flag_GPU:
            sample=self.transform(sample)
        return sample
    
