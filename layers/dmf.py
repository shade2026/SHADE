# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch import nn

from torch.autograd import Variable

# path = 'myTemp.csv'
# with open(path, encoding='utf-8') as f:
#     data = np.loadtxt(path, dtype=float, delimiter=',')

data = pd.read_csv('../dataset/Humidity/Humidity.csv').values[:, 2:].T.astype(float)

sensed_ratio = 0.1
dim_z = 21


def get_samples(data, sensed_ratio):
    m = data.shape[0]
    n = data.shape[1]
    sensed_num = int(m * sensed_ratio)
    sample_matrix = np.zeros([m, n])
    for ii in range(n):
        sample_matrix[0:sensed_num, ii] = 1
        np.random.shuffle(sample_matrix[:, ii])
    sample_data = np.multiply(data, sample_matrix)
    return sample_matrix, sample_data


sample_matrix, sample_data = get_samples(data, sensed_ratio)

batch_num = data.shape[1]
hidden_layer = 128
input_data = dim_z
output_data = data.shape[0]

z = Variable(torch.randn(batch_num, input_data), requires_grad=True)
output_target = Variable(torch.tensor(sample_data.T), requires_grad=False)

models = torch.nn.Sequential(
    torch.nn.Linear(input_data, hidden_layer),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_layer, output_data)
)

epoch_num = 10000
learning_rate = 1e-3
lmda = 0.001
loss_fn = torch.nn.MSELoss()

optimzer = torch.optim.Adam(models.parameters(), lr=learning_rate)

for epoch in range(epoch_num):
    fz = models(z)
    regularization_loss = torch.sum(z ** 2)
    for param in models.parameters():
        regularization_loss += torch.sum(param ** 2)
    loss = loss_fn(fz * torch.tensor(sample_matrix.T), output_target) + lmda * regularization_loss

    if epoch % 100 == 0:
        DMF_output = (1 - sample_matrix) * fz.T.detach().numpy() + sample_data
        RMSE = np.sqrt(np.sum((DMF_output - data) ** 2) / (data.shape[0] * data.shape[1]))
        print("Epoch={}, Loss={:.4f}, RMSE={:.4f}".format(epoch, loss.data, RMSE))

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    z.data -= learning_rate * z.grad.data
    z.grad.data.zero_()

#
# class DMF(nn.Module):
#     def __init__(self, x):
#         super(DMF, self).__init__()
#         B, T, S, C = x.shape
#         self.input_dim = 20  # dim_z
#         self.hidden_layer = 128
#         self.output_dim = S  # S
#         self.beta = 0.001
#         self.lmda = 0.001
#         self.learning_rate = 1e-3
#         self.z = nn.Parameter(torch.randn(B, T, self.input_dim))
#         self.model = nn.Sequential(
#             nn.Linear(self.input_dim, self.hidden_layer),
#             nn.Tanh(),
#             nn.Linear(self.hidden_layer, self.output_dim)
#         )
#         self.criterion = nn.MSELoss()
#
#     def get_loss(self, true, pred, mask):
#         regularization_loss = self.beta * torch.sum(self.z ** 2)
#         for param in self.model.parameters():
#             regularization_loss += self.lmda * torch.sum(param ** 2)
#         loss = self.criterion(pred * mask, true) + regularization_loss
#         return loss
#
#     def forward(self):
#         return self.model(self.z)  # B, T, S
