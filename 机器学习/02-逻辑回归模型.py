from sklearn import datasets
import numpy as np

# 导入鸾尾花数据
iris = datasets.load_iris()
# print(iris)
X = iris.data[:,[2,3]]
y = iris.target
print('Class labels:', np.unique(y))


# 训练集、测试集抽样
"""
test_size : 按7：3的概率随机抽样
random_state = 1 : 在分割数据前，内部将数据打乱洗牌
stratify_state = 1 : 返回与输入数据集的分类标签相同比例的训练和测试数据子集。
"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_test:', np.bincount(y_test))
# for i,v in zip(X_test,y_test):
#     print(i,v)


# 数据预处理：特征标准化
"""
先用StandardScaler的fit方法对训练集每个特征的 样本均值 与 标准偏差进行估算，然后调用transform方法，利用估计的参数对训练集和测试集
进行标准化，以确保可比性。
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform((X_test))

#
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined_std = np.hstack((y_train, y_test))

# ====================================================================================
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)
# 预测
y_pred = lr.predict(X_test_std)
print('【逻辑回归模型】：')
print('错误预测次数：%d次' %(y_pred != y_test).sum())
print('预测准确度：%.2f' %lr.score(X_test_std,y_test))


# 调用plot_decision_regions函数绘制模型决策区
from model_visualization import plot_decision_regions
import matplotlib.pyplot as plt

plot_decision_regions(X_combined_std, y_combined_std
                      ,classifier=lr
                      ,test_idx=range(105,150))
plt.xlabel('花瓣长度标准化')
plt.ylabel('花瓣宽度标准化')
plt.legend(loc='upper left')
plt.show()
