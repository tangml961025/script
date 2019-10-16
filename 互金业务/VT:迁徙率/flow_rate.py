"""
迁徙率是指处于某一逾期阶段的客户转到其他逾期阶段的变化情况，其可以用来预测不同逾期阶段的未来坏账损失。
比如，M2→M3，说的是从逾期阶段M2转到逾期阶段M3的比例。
通常，迁徙率用于甄别在逾期等级较高且不还款的阶段后，用户才是真正意义上的坏客户，也就是关注的坏客户坏到某一程度的比率。

逻辑：在期初前放款的订单，期末到账的各逾期等级的逾期未结清金额/期初到账的各逾期等级的订单逾期未结清金额
口径：
    a.期初自然月月初，期末自然月月末
    b.逾期等级的计算口径为：当前最大逾期天数（M0-0;M1-1~30;M2:31~60...以此类推）
    c.逾期未结清金额 也可以更换为 逾期订单数/逾期人数等
"""


import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse


# 将string类时间转化为datetime
def func_to_datetime(x:str):
    return parse(x)


# 计算两月之差
def func_mth_diff(start_mth:datetime,end_mth:datetime):
    return (end_mth.year - start_mth.year) * 12 + (end_mth.month - start_mth.month)


# 设定回溯的起始时间以及结束时间，返回期初list、期末list
def func_get_qichu_qimo_termno(start_time: str, end_time: str):
    start_mth = func_to_datetime(start_time)
    end_mth = func_to_datetime(end_time).replace(day=1)
    mth_diff = func_mth_diff(start_mth,end_mth)
    term_list = [end_mth.strftime('%Y-%m-%d 00:00:00')]

    for i in range(mth_diff):
        last_day_of_last_mth = end_mth - datetime.timedelta(days=1)
        end_mth = last_day_of_last_mth.replace(day=1)
        term_list.append(end_mth.strftime('%Y-%m-%d 00:00:00'))
    term_list.reverse()
    return term_list[:-1], term_list[1:]


# 计算输入月份的前一月，返回格式为yyyy-mm
def func_one_mth_ago(x:str):
    if not pd.isnull(x):
        first_day_of_mth = func_to_datetime(x).replace(day=1)
        last_month = (first_day_of_mth-datetime.timedelta(days=1)).strftime('%Y-%m')
        return last_month


# 计算逾期天数
def func_overdue_days(x,backpoint:str):
    if not pd.isnull(x.succ_repay_time) and x.succ_repay_time < backpoint:
        overdue_days = 0
    else:
        overdue_days = (func_to_datetime(backpoint) - func_to_datetime(x.plan_repay_time)).days
    return overdue_days


# 根据逾期天数，返回逾期等级
def func_overdue_level(overdue_days:int):
    if overdue_days < 0:
        level = None
    elif overdue_days == 0:
        level = 0
    elif overdue_days <= 30:
        level = 1
    elif overdue_days <= 60:
        level = 2
    elif overdue_days <= 90:
        level = 3
    elif overdue_days <= 120:
        level = 4
    elif overdue_days <= 150:
        level = 5
    elif overdue_days <= 180:
        level = 6
    else:
        level = 7
    return level


# 按（M1均由M0变化而来、M2均由M1变化而来...）的逻辑计算期末等级对应的期初等级
def func_qichu_level(x:int):
    return x-1 if not pd.isnull(x) else x


# 迁徙率的计算
# 简版
def func_flow_rate_simple(df_data,qichu_list:list,qimo_list:list,path:str,term_cnt:int):
    df_data_flow = pd.DataFrame(index=[0, 1, 2, 3, 4, 5, 6])

    for qichu_backpoint,qimo_backpoint in zip(qichu_list,qimo_list):
        print(qichu_backpoint, '-', qimo_backpoint)

        df_data_yifangkuan = df_data[df_data.fangkuan_time < qichu_backpoint]
        df_data_yichuzhang_qichu = df_data_yifangkuan[df_data_yifangkuan.plan_repay_time < qichu_backpoint]
        df_data_yichuzhang_qimo = df_data_yifangkuan[df_data_yifangkuan.plan_repay_time < qimo_backpoint]

        if not df_data_yichuzhang_qichu.empty:
            # 计算期初的逾期等级
            df_data_yichuzhang_qichu['qichu_overdue_days'] = df_data_yichuzhang_qichu.apply(
                lambda x: func_overdue_days(x, qichu_backpoint), axis=1)
            df_data_qichu = pd.DataFrame(df_data_yichuzhang_qichu.groupby('order_id').qichu_overdue_days.max())\
                .reset_index()
            df_data_qichu['qichu_overdue_level'] = df_data_qichu.qichu_overdue_days.apply(func_overdue_level)
            df_data_qichu_cal = pd.DataFrame(df_data_qichu.groupby('qichu_overdue_level').order_id.nunique())
            df_data_qichu_cal = df_data_qichu_cal.reindex(index=[0, 1, 2, 3, 4, 5, 6])

            # 计算期末的逾期等级
            df_data_yichuzhang_qimo['qimo_overdue_days'] = df_data_yichuzhang_qimo.apply(
                lambda x: func_overdue_days(x, qimo_backpoint), axis=1)
            df_data_qimo = pd.DataFrame(df_data_yichuzhang_qimo.groupby('order_id').qimo_overdue_days.max())\
                .reset_index()
            df_data_qimo['qimo_overdue_level'] = df_data_qimo.qimo_overdue_days.apply(func_overdue_level)
            df_data_qimo_cal = pd.DataFrame(df_data_qimo.groupby('qimo_overdue_level').order_id.nunique()).reset_index()
            df_data_qimo_cal['qichu_overdue_level'] = df_data_qimo_cal.qimo_overdue_level.apply(func_qichu_level)
            df_data_qimo_cal.set_index(['qichu_overdue_level','qimo_overdue_level'],inplace=True)
            df_data_qimo_cal = df_data_qimo_cal.reindex([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)])

            # 计算迁徙率:期末估算的期初各逾期等级人数/期初各逾期等级人数
            df_data_flow_n = round(df_data_qimo_cal/df_data_qichu_cal,4)
            df_data_flow_n.reset_index(['qimo_overdue_level'],inplace=True)
            df_data_flow_n = df_data_flow_n[['order_id']].rename(columns={'order_id': qichu_backpoint[:7]})

            # 合并
            df_data_flow = pd.concat([df_data_flow, df_data_flow_n], axis=1)
            print('ok.')
        else:
            print('pass.')

    df_data_flow.index = ["M0~M1", "M1~M2", "M2~M3", "M3~M4", "M4~M5", "M5~M6", "M6~M7+"]
    df_data_flow.index.names = ['逾期等级']
    df_data_flow.to_csv(path + 'flow_rate_simple_%sterm.csv' %term_cnt)
    return df_data_flow


# df_data
# qichu_list\qimo_list 期初期末时间列表
# path folw_rate数据储存路径
# term_cnt 订单期数，用于储存文件名称
def func_flow_rate_usual(df_data, qichu_list:list, qimo_list:list, path:str, term_cnt:int):
    df_data_flow = pd.DataFrame(index=[0, 1, 2, 3, 4, 5, 6, 7])

    for qichu_backpoint,qimo_backpoint in zip(qichu_list,qimo_list):
        print(qichu_backpoint, '-', qimo_backpoint)

        # 选取期初已放款订单
        df_data_yifangkuan = df_data[df_data.fangkuan_time < qichu_backpoint]
        # 选取期初的已出账明细
        df_data_yichuzhang_qichu = df_data_yifangkuan[df_data_yifangkuan.plan_repay_time < qichu_backpoint]
        # 选取期末已出账明细
        df_data_yichuzhang_qimo = df_data_yifangkuan[df_data_yifangkuan.plan_repay_time < qimo_backpoint]

        if not df_data_yichuzhang_qichu.empty:
            # 计算期初的逾期等级
            df_data_yichuzhang_qichu['qichu_overdue_days'] = df_data_yichuzhang_qichu.apply(
                lambda x: func_overdue_days(x, qichu_backpoint), axis=1)
            df_data_qichu = pd.DataFrame(df_data_yichuzhang_qichu.groupby('order_id').qichu_overdue_days.max())\
                .reset_index()
            df_data_qichu['qichu_overdue_level'] = df_data_qichu.qichu_overdue_days.apply(func_overdue_level)
            df_data_qichu_cal = pd.DataFrame(df_data_qichu.groupby('qichu_overdue_level').order_id.nunique())
            df_data_qichu_cal = df_data_qichu_cal.reindex(index=[0, 1, 2, 3, 4, 5, 6, 7])

            # 计算期末的逾期等级
            df_data_yichuzhang_qimo['qimo_overdue_days'] = df_data_yichuzhang_qimo.apply(
                lambda x: func_overdue_days(x, qimo_backpoint), axis=1)
            df_data_qimo = pd.DataFrame(df_data_yichuzhang_qimo.groupby('order_id').qimo_overdue_days.max())\
                .reset_index()
            df_data_qimo['qimo_overdue_level'] = df_data_qimo.qimo_overdue_days.apply(func_overdue_level)

            # 关联后计算迁徙率:期末各逾期等级变化人数/期初各逾期等级人数
            df_data_qichu_qimo = df_data_qichu.merge(df_data_qimo)
            df_data_qimo_cal = pd.DataFrame(
                df_data_qichu_qimo.groupby(['qichu_overdue_level','qimo_overdue_level']).order_id.nunique())
            df_data_qimo_cal = df_data_qimo_cal.reindex([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,7)])

            df_data_flow_n = round(df_data_qimo_cal/df_data_qichu_cal,4)
            df_data_flow_n.reset_index(['qimo_overdue_level'],inplace=True)
            df_data_flow_n = df_data_flow_n[['order_id']].rename(columns={'order_id':qichu_backpoint[:7]})

            # 合并
            df_data_flow = pd.concat([df_data_flow, df_data_flow_n], axis=1)
            print('ok.')
        else:
            print('pass.')

    df_data_flow.index = ["M0~M1", "M1~M2", "M2~M3", "M3~M4", "M4~M5", "M5~M6", "M6~M7+", "M7+~M7+"]
    df_data_flow.index.names = ['逾期等级']
    df_data_flow.to_csv(path+'flow_rate_usual_%sterm.csv' %term_cnt)
    return df_data_flow



if __name__ == "__main__":
    # nowtime = datetime.datetime.now().strftime('%Y-%m-%d')
    nowtime = '2019-08-01 00:00:00'
    qichu_time,qimo_time = func_get_qichu_qimo_termno('2018-01-01',nowtime)
    print("* 期初:%s" % qichu_time)
    print("* 期末:%s" % qimo_time)

    # path_read = "/home/data/tangmeiling/shortterm/"
    # path_save = "/home/data/tangmeiling/shortterm/flow_rate/"
    path_read = "/Users/tangmeiling/Desktop/Report/0829_短期_label定义/2-样本_sample/"
    path_save = "/Users/tangmeiling/Desktop/Report/0829_短期_label定义/迁徙率/v1/"

    df_bill = pd.read_csv(path_read+'shortterm_multiple_orderid_xlt_bill_12.csv')
    print("[1].数据读取结束.")

    # 计算期数
    df_bill_cnt = pd.DataFrame(df_bill.groupby('order_id').term_no.nunique()).reset_index()\
        .rename(columns={'term_no': 'term_cnt'})
    # 订单放款月：订单的最小应还月份-1
    df_bill_fangkuan = pd.DataFrame(df_bill.groupby('order_id').plan_repay_time.min()).reset_index()
    df_bill_fangkuan['fangkuan_time'] = df_bill_fangkuan['plan_repay_time'].apply(func_one_mth_ago)
    # 关联
    df_bill_static = df_bill_cnt.merge(df_bill_fangkuan[['order_id','fangkuan_time']])
    df_bill_orderid_detail = df_bill_static.merge(df_bill,on=['order_id'],how='inner')
    print("[2].订单期数\放款时间统计结束.")

    term = 12
    df_bill_detail = df_bill_orderid_detail[df_bill_orderid_detail.term_cnt == term]
    print("[3].选取%s期产品，开始迁徙率计算 %s" %(term,'-'*20))

    flow_rate_simple = func_flow_rate_simple(df_bill_detail, qichu_time, qimo_time, path_save, term)
    flow_rate_usual = func_flow_rate_usual(df_bill_detail,qichu_time,qimo_time,path_save,term)
    # print(flow_rate)







