from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 导入鸾尾花数据
"""
criterion:杂质度量标准
n_estimators:K值，k棵树
n_jobs:多核计算（2核）
"""
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_test:', np.bincount(y_test))


# ====================================================================================
# 随机森林
# ====================================================================================
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)


# ============================ 画图
# 调用plot_decision_regions函数绘制模型决策区
from model_visualization import plot_decision_regions
import matplotlib.pyplot as plt

X_combined_std = np.vstack((X_train, X_test))
y_combined_std = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined_std,classifier=forest,test_idx=range(105,150))
plt.xlabel('花瓣长度标准化')
plt.ylabel('花瓣宽度标准化')
plt.legend(loc='upper left')
plt.show()
