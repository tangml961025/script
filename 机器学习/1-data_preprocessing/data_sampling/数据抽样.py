import pandas as pd
import numpy as np

############################################## 准备数据
# 导入csv
data = pd.read_csv('../data/apollo_mz_feature_test_duotou_zhongan_3w.csv')
data = data[data["tag"] != 0.5]
data.tag = data.tag.astype(np.int64)
print(data.shape)
# print(data[:5])

# 提取X,y，并转化为数组
X = data.iloc[:, list(range(data.shape[1]-1))].values
y = data.iloc[:, data.shape[1]-1].values
print(X)
print(y)
print('Class labels:', np.unique(y))


############################################## 抽样
# 简单抽样
"""
test_size : 按7：3的概率随机抽样
random_state = 0 : 在分割数据前，内部将数据打乱洗牌，设定值可以复现
stratify = y : 返回与输入数据集的分类标签相同比例的训练和测试数据子集。
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify = y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# 交叉抽样
