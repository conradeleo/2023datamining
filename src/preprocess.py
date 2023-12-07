from __future__ import unicode_literals

import os
import configparser

import numpy as np
import pandas as pd
#from dask import dataframe as dd

#import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import from_numpy, log1p, cat
from torch.utils.data import TensorDataset, DataLoader
class DataProcessor:
    def __init__(self, timestep, batch_size, train=False):
        self.train = train
        self.dataset_path = '../Dataset/train_1.csv' if self.train else '../Dataset/train_2.csv'
        self.timestep = timestep
        self.batch_size = batch_size

    def __call__(self, epoch=1):
        print('\tData reading...')
        dataset = self.read_dataset()
        print(f'\t\t{dataset.shape}')
        print('\tFeatures engineering...')
        features = self.features_gen(dataset)

        print('\tDaily data generating...')
        dataX, dataY, features = self.prepare_dataset(dataset, features)

        if self.train:
            print('\t\tTrain test generating...')
            return self.train_test_gen(epoch, dataX, dataY, features)
        else:
            print('\t\tData for result generating...')
            return self.result_test_gen(epoch, dataX, dataY, features)
    
    def read_dataset(self):
        dataset = pd.read_csv(self.dataset_path, encoding='utf-8', index_col='Page')
        # Most webpages are with NaN because of not created yet
        dataset.bfill(axis=1, inplace=True)
        dataset.ffill(axis=1, inplace=True)
        # For webpages that have not been created during time period of train1
        dataset.fillna(0, inplace=True)
        dataset = dataset.astype('int')

        dataset.columns = pd.to_datetime(dataset.columns)
        return dataset
    
    def features_gen(self, dataset):
        if os.path.exists(f'../Dataset/{self.dataset_path}_features2.csv'):
            features = pd.read_csv(f'../Dataset/{self.dataset_path}_features2.csv', encoding='utf-8')
        else:
            print('\t\tparsing...')
            features = self.parse(dataset)
            print('\t\tauto-correlation...')
            autocorr = self.autocorr(dataset)
            features = pd.concat([features, autocorr], axis=1)

            features.to_csv(f'../Dataset/{self.dataset_path}_features2.csv', index=False)
        
        config = configparser.ConfigParser()
        config.read('config.ini')
        config['model_sett']['feature_size'] = str(features.shape[-1] + 2)
        with open('config.ini', 'w') as configFile:
            config.write(configFile)
        return features

    def parse(self, dataset):
        if os.path.exists(f'../Dataset/{self.dataset_path}_features1.csv'):
            features = pd.read_csv(f'../Dataset/{self.dataset_path}_features1.csv', encoding='utf-8')
            return features
        else:
            def parse_page(x):
                x = x.split('_')
                return ' '.join(x[:-3]), x[-3], x[-2], x[-1]
            name, project, access, agent = zip(*dataset.index.map(parse_page))

            #One-hot encoding of language(project), access, agent of each row
            le = LabelEncoder() #one hot encoding
            project = le.fit_transform(project)
            access = le.fit_transform(access)
            agent = le.fit_transform(agent)
            features = pd.DataFrame(list(zip(project, access, agent)), columns=['project', 'access', 'agent'])
            features.to_csv(f'../Dataset/{self.dataset_path}_features1.csv', index=False)

            return features
        
    def autocorr(self, dataset, lag=1):
        if os.path.exists(f'../Dataset/{self.dataset_path}_autocorr.csv'):
            features = pd.read_csv(f'../Dataset/{self.dataset_path}_autocorr.csv', encoding='utf-8')
            return features
        else:
            def single_autocorr(series, lag):
                # Slice the series to obtain the original and lagged series
                s1 = series[lag:]
                s2 = series[:-lag]
                ms1 = np.mean(s1)
                ms2 = np.mean(s2)

                # Calculate the deviations from the means for the original and lagged series
                ds1 = s1 - ms1
                ds2 = s2 - ms2

                # Calculate the product of the standard deviations of the original and lagged series
                divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
                # Calculate the Pearson correlation coefficient
                return np.sum(ds1 * ds2) / divider if divider != 0 else 0
            
            # Calculate year-to-year autocorrelation, resample annually, starting from July
            y_autocorr = dataset.apply(lambda row : single_autocorr(row.resample('AS-JUL').mean(), lag), axis=1)
            # Calculate quarter-to-quarter autocorrelation, resample quarterly, starting from July
            q_autocorr = dataset.apply(lambda row : single_autocorr(row.resample('QS-JUL').mean(), lag), axis=1)

            autocorr = pd.concat([y_autocorr, q_autocorr], axis=1)
            autocorr.columns = ['y_autocorr', 'q_autocorr']
            autocorr.to_csv(f'../Dataset/{self.dataset_path}_autocorr.csv', index=False)
            return autocorr

    def prepare_dataset(self, dataset, features):
        
        dataset = from_numpy(dataset.to_numpy(dtype='float32'))
        dataX = log1p(dataset).unfold(1, self.timestep, 1)[:,:-1]
        dataY = dataset[:,self.timestep:].unsqueeze(-1)
        
        features = from_numpy(features.to_numpy(dtype='float32'))
        features = features.unsqueeze(1).expand(-1, dataY.size(1), -1)
        features = cat([dataset.unsqueeze(-1)[:,self.timestep-1:-1,:], features], dim=-1)

        #print('\t\tX:', dataX.shape, 'Y:', dataY.shape, 'features', features.shape)
        #print(dataX[0, 0:2])
        #print(dataY[0, 0:2])
        #print(features[0, 0:2])
        
        return dataX, dataY, features

    def train_test_gen(self, epoch, dataX, dataY, features):
        print('\t\tX:', dataX.shape, 'Y:', dataY.shape, 'features', features.shape)
        # Split the data into training, validation, and test sets
        for i in range(epoch):
            X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(dataX, dataY, features, test_size=0.2, shuffle=True)

            # Create TensorDatasets
            train_dataset = TensorDataset(X_train, y_train, f_train)
            test_dataset = TensorDataset(X_test, y_test, f_test)

            # Create DataLoader
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

            yield i, train_loader, test_loader

    def result_test_gen(self, epoch, dataX, dataY, features):
        for i in range(epoch):
            dataX = dataX[:,550-self.timestep:,:]
            dataY = dataY[:,550-self.timestep:,:]
            features = features[:, 550-self.timestep:, :]
            print('\t\tX:', dataX.shape, 'Y:', dataY.shape, 'features', features.shape)

            # Create TensorDatasets
            test_dataset = TensorDataset(dataX, dataY, features)
            # Create DataLoader
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

            yield i, test_loader