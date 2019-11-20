import json
import time
import re
import pandas as pd
import os


# 把json平铺开
def flatten(d, parent_key=''):
    items = []
    for k, v in d.items():
        try:
            items.extend(flatten(v, '%s' % k).items())
        except AttributeError:
            items.append(('%s' % k, v))
    return dict(items)


# 获取json数据
def get_online_data(online_path):
    with open(online_path, 'r') as f:
        data = json.load(f)
    data = data['data']
    result = flatten(data)
    return result


# 判断是否是数值类型
def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


# 线上线下对比
def compare_result():
    for i in df.index:
        if is_number(str(df.loc[i, 'offline'])) and is_number(str(df.loc[i, 'online'])):
            if float(df.loc[i, 'online']) == float(df.loc[i, 'offline']):
                df.loc[i, 'compare'] = ''
            else:
                df.loc[i, 'compare'] = 'error'
        else:
            if df.loc[i, 'online'] == df.loc[i, 'offline']:
                df.loc[i, 'compare'] = ''
            else:
                df.loc[i, 'compare'] = 'error'


# 插入线上线下结果
def insert_result(key):
    if key != 'name' :
        # 线下结果
        if str(offline_all.loc[name, key]) == '-99998.0' :
            offline_result = '-99998'
        elif str(offline_all.loc[name, key]) == '-99999.0' :
            offline_result = '-99999'
        else :
            offline_result = str(offline_all.loc[name, key])

        df.loc[key, 'offline'] = offline_result

        # 线上结果
        try :
            df.loc[key, 'online'] = str(result[key])
        except :
            df.loc[key, 'online'] = '无'

    else:
        df.loc[key, 'offline'] = name
        df.loc[key, 'online'] = name


if __name__ == '__main__':
    # 生成对比结果
    write = pd.ExcelWriter('/Users/moxie/Desktop/mz_all_compare_tml_0719.xlsx')
    offline_all = pd.read_csv('offline.csv')
    compare_index = offline_all.columns.values.tolist()
    df_all = pd.DataFrame(index=compare_index, columns=['异常'])
    offline_all = offline_all.set_index('name')

    online_names = []
    n = 0
    file_list = os.listdir('/Users/moxie/PycharmProjects/peiqi_new/online/')
    print(file_list)
    
    # 批量修改线上txt结果为json
    for file in file_list :
        online_names.append(str(str(file).split('+')[0]))
        old_name = '/Users/moxie/PycharmProjects/peiqi_new/online/' + str(file)
        new_name = '/Users/moxie/PycharmProjects/peiqi_new/online/' + str(file.split('+')[0]) + '.json'
        print(old_name)
        print(new_name)
        os.rename(old_name, new_name)
        n += 1

    for file in file_list:
        online_names.append((file.split('+')[0]).split('.')[0])
    name_list = offline_all.index.tolist()
    print(name_list)
    for name in name_list:
        print(name)
        if name in online_names:
            domain = os.path.abspath('/Users/moxie/PycharmProjects/peiqi_new/online')
            online_path = os.path.join(domain, name+'.json')
            result = get_online_data(online_path)
            df = pd.DataFrame(index=compare_index,columns=['offline', 'online', 'compare'])
            for feature in df.index:
                insert_result(feature)
            compare_result()
            df_all = pd.concat([df_all, df], axis=1)
        else:
            pass
    df_all.to_excel(write, sheet_name='样本', index=True)
    write.save()