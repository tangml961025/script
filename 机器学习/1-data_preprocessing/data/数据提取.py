import pandas as pd

# 数据读取
data = pd.read_csv("/Users/tangmeiling/Documents/1-魔蝎/work/Analysis/0-analysis_data/z-apollo_mz_all_duotou_tenant_desc_zhongan_201906_feature_all_1.csv")
print(data.shape)
print(data[:1])
print(data['tag'].value_counts())

# 特征筛选
# 删除缺失率过高的指标
drop_isnull = []
isnull_rate = []
iszero_rate = []
for col in data.columns:
    isnull_rate_i = round(data[data[col].isnull()].shape[0]/data.shape[0],4)
    iszero_rate_i = round(data[data[col]==0].shape[0]/data.shape[0],4)
    isnull_rate.append(isnull_rate_i)
    iszero_rate.append(iszero_rate_i)
    if (isnull_rate_i > 0.7) | (iszero_rate_i > 0.7):
        drop_isnull.append(col)

# 缺失率
print("最大缺失率：", max(isnull_rate))
# 0值
iszero_rate_la = [x for x in iszero_rate if x > 0.7]
print("0值百分比>70%的特征数：",len(iszero_rate_la))
# 删除特征
data_drop = data.drop(drop_isnull,axis=1)
print(data_drop.shape)

# 选取只180d特征
col_180d = []
for col in data_drop.columns:
    if (col in ['user_id', 'org_count','tag']) | ("180d" in col.split('_')):
        col_180d.append(col)
print("剩余特征数：", len(col_180d))
data_180d = data_drop.loc[:,col_180d]

# 选取tag=0\0.5\1的数据各1w条
data_index = []
data_0_index = list(data_180d[data_180d['tag']==0][:30000].index)
data_1_index = list(data_180d[data_180d['tag']==0.5][:10000].index)
data_2_index = list(data_180d[data_180d['tag']==1][:10000].index)
# 合并
data_index.extend(data_0_index)
data_index.extend(data_1_index)
data_index.extend(data_2_index)
print(len(data_index))
# 提取
data_3w = data_180d.loc[data_index,]
# 导出
data_3w.to_csv('./apollo_mz_feature_test_duotou_zhongan_3w.csv',index=False)


