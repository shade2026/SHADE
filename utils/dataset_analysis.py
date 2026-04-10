import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['axes.unicode_minus'] = False  # 禁用Unicode减号
import torch

# 比较时域正常数据和异常数据
index=20
L=8000
k=5

data = pd.read_csv('dataset/TrafficFlow/TrafficFlow_01.csv')
cols = list(data.columns)
cols.remove('time')
cols.remove('week')
data=data[cols].values
x = torch.tensor(data[:, index],dtype=torch.float)[:L]

data1 = pd.read_csv('dataset/TrafficFlow/TrafficFlow.csv')
cols = list(data1.columns)
cols.remove('time')
cols.remove('week')
data1=data1[cols].values
x1 = torch.tensor(data1[:, index],dtype=torch.float)[:L]

xf = torch.fft.rfft(x, dim=0)  # B, L/2, S, D
frequency_amplitudes = abs(xf)  # B, L/2, S, D
# 屏蔽直流frequency
frequency_amplitudes[0] = 0  # B, L/2, S, D
# extract topk part
top_indices = torch.topk(frequency_amplitudes, k, dim=0).indices  # 取索引, (B, k, S, D)

mask = torch.zeros_like(xf)
mask.scatter_(0, top_indices, 1)
xf_filtered = xf * mask

main_frequency = torch.fft.irfft(xf_filtered, n=x.shape[0], dim=0)  # B, L, S, D
res = x - main_frequency  # B, L, S, D
plt.figure(figsize=(16, 8))
plt.plot(x, label='x')
# plt.plot(main_frequency, label='main')
# plt.plot(res, label='res')
plt.plot(x1, label='x1')
plt.legend()
plt.show()




# 频谱分析
index=20
L=8000
k=5
data = pd.read_csv('dataset/TrafficFlow/TrafficFlow_01.csv')
cols = list(data.columns)
cols.remove('time')
cols.remove('week')
data=data[cols].values
x = torch.tensor(data[:, index],dtype=torch.float)[:L]
data = pd.read_csv('dataset/TrafficFlow/TrafficFlow.csv')
cols = list(data.columns)
cols.remove('time')
cols.remove('week')
data=data[cols].values
y = torch.tensor(data[:, index],dtype=torch.float)[:L]
# data = pd.read_csv('dataset/Humidity/Humidity_0124.csv')
# cols = list(data.columns)
# cols.remove('time')
# cols.remove('week')
# data=data[cols].values
# z = torch.tensor(data[:, index],dtype=torch.float)[:L]

xf = torch.fft.rfft(x, dim=0)  # B, L/2, S, D
frequency_amplitudes = abs(xf)  # B, L/2, S, D
# 屏蔽直流frequency
frequency_amplitudes[0] = 0  # B, L/2, S, D

yf = torch.fft.rfft(y, dim=0)  # B, L/2, S, D
frequency_amplitudes1 = abs(yf)  # B, L/2, S, D
# 屏蔽直流frequency
frequency_amplitudes1[0] = 0  # B, L/2, S, D
#
# zf = torch.fft.rfft(z, dim=0)  # B, L/2, S, D
# frequency_amplitudes2 = abs(zf)  # B, L/2, S, D
# # 屏蔽直流frequency
# frequency_amplitudes2[0] = 0  # B, L/2, S, D

plt.figure(figsize=(16, 8))
plt.plot(frequency_amplitudes, label='frequency_amplitudes_anomalous_01')
plt.plot(frequency_amplitudes1, label='frequency_amplitudes_normal')
# plt.plot(frequency_amplitudes2, label='frequency_amplitudes_anomalous_0124')
plt.legend(fontsize='x-large')
plt.show()