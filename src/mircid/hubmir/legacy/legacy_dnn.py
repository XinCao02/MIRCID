# -*- coding: UTF-8 -*-
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.nn import init
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
torch.set_default_tensor_type(torch.FloatTensor)

#print(torch.__version__)

#training device
device = torch.device("cuda")

# show all results in each data units
from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = "all" 

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=0.3),  # dropout
            nn.Linear(hidden_size, hidden_size),
            
            #nn.ReLU(),
            #nn.BatchNorm1d(hidden_size),
            #nn.Dropout(p=0.3),
            
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=0.3),
            
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, x):
        x = self.model(x)
        return x

# define a loss function
loss_fn = torch.nn.MSELoss().to(device)

def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(2 * loss_fn(clipped_preds.log(),labels.log()).mean())
    return rmse.item()
#log_rmse=log_rmse.to(device)

#network application
net=torch.load("../files/step4_DNN_miRNA199.pth",map_location=torch.device('cpu'))
net.eval()
#net = net.to("cpu")

input_csv = sys.argv[1]+'extracted_input.csv'
X_test_P = pd.read_csv(sys.argv[1],index_col=0) 
X_test_Tensor = torch.tensor(X_test_P.values,dtype=torch.float)
y_test_P = pd.read_csv("../files/1298miRNA.csv", index_col=0)
y_test_out = pd.DataFrame(net(X_test_Tensor).detach().numpy())
y_test_out.columns = y_test_P.columns

#result extraction
top414 = pd.read_csv('../files/top414.csv',names=['names'],header=0)
top_miRNA=list(top414.loc[:,'names'])
y_test_out_top414 = y_test_out.loc[:,top_miRNA]
y_test_out_top414.index = X_test_P.index
transpose=y_test_out_top414.transpose()
output_dir = str(sys.argv[1])+'predicted_miRNA_output.csv'
transpose.to_csv(output_dir) 