#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import datetime
from dateutil import parser

# df_bill_all 还款明细表
bill = pd.read_csv('/home/data/tangmeiling/shortterm/shortterm_multiple_orderid_xlt_bill_12.csv')
##############################################################################################################################算逾期
# 增加订单期数字段
test = (pd.DataFrame(bill['bill_id'].groupby(bill['order_id']).count()).reset_index()).rename(columns={'bill_id': 'term_cnt'})
bill = pd.merge(bill,test,on='order_id',how='inner')
# 增加订单放款月字段（订单的最小应还月份-1）
def one_mth_ago(x):
    if str(x) != 'nan':
        first_day_of_mth = parser.parse(x).replace(day=1)
        last_month = (first_day_of_mth - datetime.timedelta(days=1)).strftime('%Y-%m')
    return last_month
test = pd.DataFrame(bill['plan_repay_time'].groupby(bill['order_id']).min()).reset_index()
test['first_loantime'] = test['plan_repay_time'].apply(one_mth_ago)
bill = pd.merge(bill,test[['order_id','first_loantime']],on='order_id',how='inner')

def get_vt(bill, backpoint):
    df_vt = pd.DataFrame(index=backpoint[:-2],columns=list(range(1,len(backpoint))))
    for time in backpoint[2:]:
        backpoint_date = parser.parse(time + '-01').strftime('%Y-%m-%d 00:00:00')
        # 已出账+未结清 = 当前逾期
        bill_yichuzhang = bill[bill['plan_repay_time'] < backpoint_date]
        bill_yichuzhang['yuqi'] = bill_yichuzhang.apply(lambda x:1 if pd.isnull(x.succ_repay_time) or x.succ_repay_time>=backpoint_date else 0,axis=1)

        # 计算vt
        bill_yichuzhang_all = pd.DataFrame(bill_yichuzhang.groupby('first_loantime').order_id.count())
        bill_yichuzhang_yuqi = pd.DataFrame(bill_yichuzhang[bill_yichuzhang.yuqi==1].groupby('first_loantime').order_id.count())
        df_bill_vt_value = bill_yichuzhang_yuqi/bill_yichuzhang_all

        # df_bill_vt_value = pd.merge(bill_yichuzhang_all,bill_yichuzhang_yuqi,on='first_loantime', how='left')
        # df_bill_vt_value['overdue_ratio'] = round(df_bill_vt_value['order_id_y'] / df_bill_vt_value['order_id_x'], 4)

        # 将所有结果汇总
        for loan_time in df_bill_vt_value.index:
            df_vt.loc[loan_time,month_differ(time,loan_time)] = round(df_bill_vt_value.loc[loan_time,'order_id'],4)
    return df_vt

# main
bill12 = bill[bill.term_cnt==12]
#最小贷款月份201801
df = get_vt(bill12, ['2018-01','2018-02','2018-03', '2018-04', '2018-05', '2018-06', '2018-07',
                   '2018-08', '2018-09', '2018-10', '2018-11', '2018-12', '2019-01', '2019-02',
                   '2019-03', '2019-04', '2019-05', '2019-06','2019-07'])


##############################################################################################################################算DPD30
import calendar
from dateutil import parser
import datetime
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth",1000)

bill = pd.read_csv('shortterm_multiple_orderid_xlt_bill_12.csv')
# 增加订单期数字段
test = (pd.DataFrame(bill['bill_id'].groupby(bill['order_id']).count()).reset_index()).rename(columns={'bill_id': 'term_cnt'})
bill = pd.merge(bill,test,on='order_id',how='inner')
# 增加订单放款月字段（订单的最小应还月份-1）
def one_mth_ago(x):
    if str(x) != 'nan':
        first_day_of_mth = parser.parse(x).replace(day=1)
        last_month = (first_day_of_mth - datetime.timedelta(days=1)).strftime('%Y-%m')
    return last_month
test = pd.DataFrame(bill['plan_repay_time'].groupby(bill['order_id']).min()).reset_index()
test['first_loantime'] = test['plan_repay_time'].apply(one_mth_ago)
bill = pd.merge(bill,test[['order_id','first_loantime']],on='order_id',how='inner')

def string_toDatetime(st):
    return datetime.datetime.strptime(st, "%Y-%m-%d %H:%M:%S")

def func_overduedays(x,end_time):
    if not pd.isnull(x.succ_repay_time) and x.succ_repay_time<=end_time:
        cut = 0
    else:
        cut = (string_toDatetime(end_time)-string_toDatetime(x.plan_repay_time)).days
    return cut

def get_start_end_time(date):
    year, month = str(date).split('-')[0], str(date).split('-')[1]
    end = calendar.monthrange(int(year), int(month))[1]
    start_time = '%s-%s-01 00:00:00' % (year, month)
    end_time = '%s-%s-%s 23:59:59' % (year, month, end)
    return start_time,end_time

def month_differ(str1,str2):
    year1=datetime.datetime.strptime(str1,"%Y-%m").year
    year2=datetime.datetime.strptime(str2,"%Y-%m").year
    month1=datetime.datetime.strptime(str1,"%Y-%m").month
    month2=datetime.datetime.strptime(str2,"%Y-%m").month
    num=(year1-year2)*12+(month1-month2)
    return num

def get_vt(bill, backpoint):
    vt_30 = pd.DataFrame(index=backpoint[:-2],columns=range(len(backpoint)))
    vt_60 = pd.DataFrame(index=backpoint[:-2], columns=range(len(backpoint)))
    vt_90 = pd.DataFrame(index=backpoint[:-2], columns=range(len(backpoint)))
    for time in backpoint[1:]:
        start_time, end_time = get_start_end_time(time)

        ## 已出账+未结清 = 当前逾期
        bill_yichuzhang = bill[bill['plan_repay_time'] <= end_time]#已出账
        bill_yichuzhang['cut'] = bill_yichuzhang.apply(lambda x:func_overduedays(x,end_time),axis=1)

        ## 计算vt
        bill_yichuzhang_all = pd.DataFrame(bill_yichuzhang.groupby('first_loantime').order_id.nunique())
        # 30
        bill_yichuzhang_30 = pd.DataFrame(bill_yichuzhang[bill_yichuzhang.cut > 30].groupby('first_loantime').order_id.nunique())
        vt_30_value = bill_yichuzhang_30/bill_yichuzhang_all
        # 60
        bill_yichuzhang_60 = pd.DataFrame(bill_yichuzhang[bill_yichuzhang.cut > 60].groupby('first_loantime').order_id.nunique())
        vt_60_value = bill_yichuzhang_60 / bill_yichuzhang_all
        # 90
        bill_yichuzhang_90 = pd.DataFrame(bill_yichuzhang[bill_yichuzhang.cut > 90].groupby('first_loantime').order_id.nunique())
        vt_90_value = bill_yichuzhang_90 / bill_yichuzhang_all

        # 将所有结果汇总
        for loan_time in vt_30_value.index:
            vt_30.loc[loan_time,month_differ(time,loan_time)] = round(vt_30_value.loc[loan_time,'order_id'],4)
        for loan_time in vt_60_value.index:
            vt_60.loc[loan_time,month_differ(time,loan_time)] = round(vt_60_value.loc[loan_time,'order_id'],4)
        for loan_time in vt_90_value.index:
            vt_90.loc[loan_time,month_differ(time,loan_time)] = round(vt_90_value.loc[loan_time,'order_id'],4)
    return vt_30,vt_60,vt_90


month_list = ['2018-01','2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12','2019-01',
        '2019-02','2019-03','2019-04','2019-05','2019-06']


data = bill[bill.term_cnt==3]
#最小贷款月份201801
vt_30,vt_60,vt_90 = get_vt(data, month_list)
