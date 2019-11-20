from sklearn import datasets
import numpy as np

# 导入鸾尾花数据
iris = datasets.load_iris()
# print(iris)
X = iris.data[:,[2,3]]
y = iris.target
print('Class labels:', np.unique(y))
print("X:\n%s" %X)
print("y:\n{}".format(y))


# 训练集、测试集抽样
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


######################################### 数据预处理
# 特征标准化
"""
先用StandardScaler的fit方法对训练集每个特征的 样本均值 与 标准偏差进行估算，然后调用transform方法，利用估计的参数对训练集和测试集
进行标准化，以确保可比性。
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform((X_test))

