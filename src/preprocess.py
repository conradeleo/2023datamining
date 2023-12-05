from __future__ import unicode_literals

import os
import csv
import ast
import time
import math
from datetime import datetime

import numpy as np
import pandas as pd
from dask import dataframe as dd

#import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import from_numpy, log1p, cat
from torch.utils.data import TensorDataset, DataLoader

from config import train1_sequence_len, train2_sequence_len

class DataProcessor:
    def __init__(self, dataset_path, timestep, batch_size):
        self.train_length = train1_sequence_len
        self.test_length = train2_sequence_len-train1_sequence_len
        self.dataset_path = dataset_path
        #dataset = dataset.reset_index(drop=True).set_index('Page')
        self.timestep = timestep
        self.batch_size = batch_size

    def __call__(self):
        print('\tData reading...')
        dataset = pd.read_csv(f'../Dataset/{self.dataset_path}', encoding='utf-8', index_col='Page').fillna(0).astype('int')
        print('\tFeatures parsing...')
        features = self.parse(dataset)
        self.autocorr(dataset, 365)

        print('\tDaily data generating...')

        X_train, y_train, f_train, X_val, y_val, f_val, X_test, y_test, f_test = self.daily_data_generator(dataset, features)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(X_train, y_train, f_train)
        val_dataset = TensorDataset(X_val, y_val, f_val)
        test_dataset = TensorDataset(X_test, y_test, f_test)

        # Create DataLoader
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def parse(self, dataset):
        if os.path.exists(f'../Dataset/{self.dataset_path}_features.csv'):
            features = pd.read_csv(f'../Dataset/{self.dataset_path}_features.csv', encoding='utf-8', header=None)
            return features
        else:
            def parse_page(x):
                x = x.split('_')
                return ' '.join(x[:-3]), x[-3], x[-2], x[-1]
            name, project, access, agent = zip(*dataset.index.map(parse_page))
            
            #log_mean_values = dataset.apply(lambda row: int(np.log(np.mean(row)+1)+1), axis=1)

            #One-hot encoding of language(project), access, agent, logmean_of_each_row
            le = LabelEncoder() #one hot encoding
            project = le.fit_transform(project)
            access = le.fit_transform(access)
            agent = le.fit_transform(agent)
            feature_list = zip(project, access, agent)
            with open(f"../Dataset/{self.dataset_path}_features.csv", "w", encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(feature_list)

            return feature_list
        
    def autocorr(self, dataset, lag):
        autocorr_result = dataset.apply(lambda x: x.autocorr(lag=lag))  # Assuming daily frequency, adjust as needed
        print(autocorr_result.head())
        
    def daily_data_generator(self, dataset, feature):
        '''
        shape = (len(concatnated_dataset), len(concatnated_dataset.columns), len(concatnated_dataset.iloc[0,0]))
        arrays_list = [np.array(t) for t in concatnated_dataset.to_numpy().flatten()]
        concatnated_dataset = np.array(arrays_list).reshape(shape)
        '''
        data = from_numpy(dataset.to_numpy(dtype='float32'))
        dataY = data[:, -(train2_sequence_len - self.timestep + 1):].unsqueeze(-1)
        dataX = data.unfold(1, self.timestep, 1).unsqueeze(-1)
        dataX_transformed = log1p(dataX)
        dataX = cat((dataX_transformed, dataX), dim=-1).flatten(start_dim=2)
        
        feature = from_numpy(feature.to_numpy(dtype='float32'))
        
        print('X:', dataX.shape, 'Y:', dataY.shape)
        print(dataX[0, 0:2])
        print(dataY[0, 0:2])
        #获取训练集大小
        train_size = self.train_length - self.timestep

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(dataX, dataY, feature, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(X_temp, y_temp, f_temp, test_size=0.5, random_state=42)
        print('x_train:', X_train.shape, 'y_train:', y_train.shape)
        print('x_val:', X_val.shape, 'y_val:', y_test.shape)
        print('x_test:', X_test.shape, 'y_test:', y_test.shape)

        return X_train, y_train, f_train, X_val, y_val, f_val, X_test, y_test, f_test