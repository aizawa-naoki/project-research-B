import torch  # 基本モジュール
from torch.autograd import Variable  # 自動微分用
import torch.nn as nn  # ネットワーク構築用
import torch.optim as optim  # 最適化関数
import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data  # データセット読み込み関連
import pandas as pd
import numpy as np
from util import Net, RESTDataset

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


#-------------------- start init ------------------------#
net = Net(input_size=768, hidden_size=100, output_size=12)
net = net.to(device)
shuffle = True
batch_size = 40
print_per = 300
criterion = nn.modules.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#--------------------  end init  ------------------------#

####################### data load & convert #######################
dataset = RESTDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=40, shuffle=True)

running_loss = 0.0

for epoch in range(1500):
    # print loss while running
    if epoch % print_per == (print_per - 1):
        print("loop: %4d loss: %.3f" %
              (epoch + 1, (running_loss / print_per)))
    running_loss = 0.0
    for batch in dataloader:
        mini_batch_x, mini_batch_y = batch

        # forward
        optimizer.zero_grad()
        predicts = net(mini_batch_x.float().to(device))
        loss = torch.mean(
            criterion(predicts, mini_batch_y.float().to(device)), dim=0)
        # backward
        loss.backward()
        # optimize
        optimizer.step()

        running_loss += loss.sum()


####################### evaluation #######################
df_val_x = pd.read_csv("../data/REST_test_x_vec.csv", header=None)
df_val_y = pd.read_csv("../data/REST_test_y.csv", header=None)
val_x = torch.from_numpy(df_val_x.iloc[:, 1:].values).float().to(device)
val_y = df_val_y.iloc[:, 1:].values
print("model-mode = evaluation")
net.eval()
with torch.no_grad():
    predict = net(val_x)
predict = predict.cpu()
predict = predict.numpy()
result = predict.round()
result = result * 1.2
result = result.astype("int8")
val_y = val_y * 1.2
val_y = val_y.astype("int8")

TP = np.count_nonzero((result == 1) & (val_y == 1))
FP = np.count_nonzero((result == 1) & (val_y == 0))
FN = np.count_nonzero((result == 0) & (val_y == 1))
TN = np.count_nonzero((result == 0) & (val_y == 0))


accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP + 1e-10)
recall = TP / (TP + FN + 1e-10)
F1 = 2 * precision * recall / (precision + recall + 1e-10)
print("TP:%4d, FP:%4d" % (TP, FP))
print("FN:%4d, TN:%4d" % (FN, TN))
print("accuracy:%5f\nprecision:%5f\nrecall:%5f\nF1:%5f\n" %
      (accuracy, precision, recall, F1))
