import os
import pandas as pd
import torch
import

os.makedirs(os.path.join('..','data'),exist_ok=True)# 在上级目录创建data文件夹
data_file = os.path.join('..','data','house_tiny.csv')
with open(data_file, 'w') as f1:
    f1.write('NumRooms,Alley,Price\n')
    f1.write('NA,Pave,127500\n')
    f1.write('2,NA,106000\n')
    f1.write('4,NA,178100\n')
    f1.write('NA,NA,140000\n')
# with open(data_file, 'r') as f1:
#     for line in f1.readlines():
#         print(line)

data = pd.read_csv(data_file)# 如果NA前加空格，就显示NA，不加空格就NAN
print(data)

inputs, outputs = data.iloc[:,0:2],data.iloc[:,2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)# 相当于独立热编码
print(inputs)

X,y = torch.Tensor(inputs.values),torch.Tensor(outputs.values)
print(X,y)

