import torch
import os
import pandas as pd
import math
import numpy as np
from deepcore.datasets.script import dataloader, utility, earlystopping, opt
from sklearn import preprocessing
import torch.utils as utils

def Traffic(dataset:str, gso_type:str, graph_conv_type:str, device, batch_size:int, n_his:int, n_pred:int):
    adj, n_vertex = dataloader.load_adj(dataset)
    gso = utility.calc_gso(adj, gso_type)
    if graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path,dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, n_his, n_pred, device)
    x_val, y_val = dataloader.data_transform(val, n_his, n_pred, device)
    x_test, y_test = dataloader.data_transform(test, n_his, n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    val_data = utils.data.TensorDataset(x_val, y_val)
    test_data = utils.data.TensorDataset(x_test, y_test)

    return gso, n_vertex, zscore, train_data, val_data, test_data

def MetrLa(gso_type:str, graph_conv_type:str, device, batch_size:int, n_his:int, n_pred:int):
    return Traffic('metr-la', gso_type, graph_conv_type, device, batch_size, n_his, n_pred)

def PemsBay(gso_type:str, graph_conv_type:str, device, batch_size:int, n_his:int, n_pred:int):
    return Traffic('pems-bay', gso_type, graph_conv_type, device, batch_size, n_his, n_pred)

def Pemsd7M(gso_type:str, graph_conv_type:str, device, batch_size:int, n_his:int, n_pred:int):
    return Traffic('pemsd7-m', gso_type, graph_conv_type, device, batch_size, n_his, n_pred)