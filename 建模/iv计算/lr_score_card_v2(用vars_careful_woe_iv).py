# -*- coding:utf-8 -*-

# time : 2018-06-20
# author : baron
# python 3.5 

import numpy as np
import pandas as pd

from sklearn.externals import joblib
import os
from os.path import abspath
from os.path import join
from os.path import dirname
import statsmodels.api as sm
import re

import sys
from imp import reload


# 过滤掉警告
import warnings
warnings.filterwarnings("ignore")



########################################################################
############# 变量粗分箱
########################################################################


# updata 2018-06
def vars_rough_woe_iv(data, feature,path='.',sub_path='rough_woe',ifsave=False,bins=11):
    """

    :param data: DataFrame
    :param feature: cols include string type and numerical type 
    :param path: the path to save woe.xls
    :param ifsave: weather save _woe.xls
    :param bins: 分箱数目
    :return: the woe dict of all vars and iv of the vars
    """
    cal = pd.DataFrame(index=['iv', 'cut_count', 'monotone'])
    woe_dict = {}
    for i in feature:
        print (i)
#        print('='*10)
        if data[i].dtype == object:
#            print('----- the var {0} is object'.format(np.str(i)))
            data[i + '_cut'] = data[i]
            if data[i + '_cut'].isnull().sum() > 0 :
#                print(' ----- the var {0} for bins include null'.format(np.str(i)))
                df1 = data[data[i + '_cut'].isnull()][[i + '_cut','tag']]
                total = len(df1)
                bad = df1['tag'].sum()
                good = sum(df1['tag'] == 0)
                # 坏用户占坏用户总数的比例
                bad_pcnt = 1.0 * bad / data['tag'].sum()
                good_pcnt = 1.0 * good / sum(data['tag'] == 0)
                odds = 1.0*good / bad
                total_pct = round(1.0*total / data.shape[0],2)
                woe = np.log(good_pcnt * 1.0 / bad_pcnt) if good_pcnt != 0 and bad_pcnt !=0 else 0
                iv_null = (good_pcnt-bad_pcnt)*np.log(good_pcnt*1.0/bad_pcnt) if bad_pcnt !=0 else 0
                woe_df_null = [np.nan, total, bad, good, good_pcnt, bad_pcnt, woe,odds,total_pct,iv_null]
                df_notnull = data

            else:
                df_notnull = data


            woe_df, WOE_dict, iv, std = calcwoe(df_notnull, i + '_cut')
            # 该单调性的计算不包含null值对应的woe在内，故在合并woe之前计算
            mono = mono_or_u(woe_df['WOE'])
            whn =  0 
            try:
                # 当没有缺失时，该print产生异常，不进行此部分处理
#                print(woe_df_null)
                last_index = woe_df.shape[0] + 1
                woe_df.ix[last_index,i+ '_cut'] = np.nan
                woe_df.ix[last_index,'total'] = woe_df_null[1]
                woe_df.ix[last_index,'bad'] = woe_df_null[2]
                woe_df.ix[last_index,'good'] = woe_df_null[3]
                woe_df.ix[last_index,'good_pcnt'] = woe_df_null[4]
                woe_df.ix[last_index,'bad_pcnt'] = woe_df_null[5]
                woe_df.ix[last_index,'WOE'] = woe_df_null[6]
                woe_df.ix[last_index,'odds'] = woe_df_null[7]
                woe_df.ix[last_index,'total_pct'] = woe_df_null[8]
                woe_df.ix[last_index,'sub_iv'] = woe_df_null[9]

                iv = iv + iv_null
                woe_df['iv'] = iv
                whn = 1
            except NameError:
                pass
            print (woe_df_null)
            WOE_dict_new = {}
            for key in WOE_dict.keys():
                    WOE_dict_new[key] = WOE_dict[key]['WOE']
                    

            WOE_dict_new_ = WOE_dict_new.copy()
            if whn == 1:
                WOE_dict_new[np.nan] = woe_df_null[6]
                tmp = woe_df_null[6]
                WOE_dict_new_['NaN'] = tmp
    
          
            data[i + '_cut'].fillna('NaN',inplace=True)
            data[i + '_woe'] = data[i + '_cut'].map(lambda x:WOE_dict_new_[x]).astype(np.float64)
            data[i + '_cut'].replace('NaN',np.nan,inplace=True)
    
            cut_cnt = len(woe_df)
            iv_mono = [iv, cut_cnt, mono]
            
            
            tmp = data[i + '_cut'].value_counts(dropna=False)
            tmp.sort_index(ascending=True,inplace=True)
            cutpoint = list(tmp.index)
            cutpoint = [x for x in cutpoint if x != 'NaN']

        else:

            data[i + '_cut'], cutpoint, woe_df, iv_mono,data[i + '_woe'],WOE_dict_new = var_rough_bins(data,i,bins)
            try:
                data.ix[data[i + '_woe'].isnull(),i + '_woe'] = WOE_dict_new[np.nan]
            except KeyError:
                pass
            #elif (data[i + '_woe']==-99999).sum() > 0:
            #    data.ix[data[i + '_woe']==-99999,i + '_woe'] = WOE_dict_new['-99999']

        # k = ks(data, i + '_cut')
        # maxbin = woe_df['total'].max() / len(data)
        cal[i] = iv_mono
        
        woe_df['cut_point'] = [cutpoint]*len(woe_df) 
        woe_dict[i] = woe_df[[i+'_cut','cut_point','total','good','bad','good_pcnt','bad_pcnt','WOE','odds','total_pct','sub_iv','iv']]
        if ifsave == True:
            save_var_bins_woe_iv(woe_dict[i],cal[i],path,sub_path)
    return woe_dict,cal







def var_rough_bins(df,col,bins=11):
#    print('the var for bins is %s' % col)
    col_cut = col + '_cut'
    if df[col].isnull().sum() > 0 :
#        print(' ----- the var for bins include null')
        df1 = df[df[col].isnull()][[col,'tag']]
        total = len(df1)
        bad = df1['tag'].sum()
        good = sum(df1['tag'] == 0)
        bad_pcnt = 1.0 * bad / df['tag'].sum()
        good_pcnt = 1.0 * good / sum(df['tag'] == 0)
        # good_rate = good / total
        woe = np.log(good_pcnt * 1.0 / bad_pcnt) if good_pcnt != 0 and bad_pcnt !=0 else 0
        iv_null = (good_pcnt-bad_pcnt)*np.log(good_pcnt*1.0/bad_pcnt) if bad_pcnt !=0 else 0
        odds = 1.0*good/bad
        total_pct = 1.0*total/df.shape[0]
        woe_df_null = [np.nan, total, bad, good, good_pcnt, bad_pcnt, woe,odds,total_pct,iv_null]
        df_notnull = df
#改
        
    elif (df[col]==-99999).sum() > 0 :
#        print(' ----- the var for bins include null')
        df1 = df[(df[col]==-99999)][[col,'tag']]
        total2 = len(df1)
        bad2 = df1['tag'].sum()
        good2 = sum(df1['tag'] == 0)
        bad_pcnt2 = 1.0 * bad2 / df['tag'].sum()
        good_pcnt2 = 1.0 * good2 / sum(df['tag'] == 0)
        # good_rate = good / total
        woe2 = np.log(good_pcnt2 * 1.0 / bad_pcnt2) if good_pcnt2 != 0 and bad_pcnt2 !=0 else 0
        iv_sp = (good_pcnt2-bad_pcnt2)*np.log(good_pcnt2*1.0/bad_pcnt2) if bad_pcnt2 !=0 else 0
        odds2 = 1.0*good2/bad2
        total_pct2 = 1.0*total2/df.shape[0]
        woe_df_sp = [-99999, total2, bad2, good2, good_pcnt2, bad_pcnt2, woe2,odds2,total_pct2,iv_sp]
        df_notnull = df
        
    else:
        df_notnull = df

    cutOffPoints = list(set(df_notnull[col].quantile(np.linspace(0.05,0.95,num=bins))))
    cutOffPoints.append(-np.inf)
    cutOffPoints.append(np.inf)
    cutOffPoints.sort()

    df_notnull[col_cut],tmp2 = pd.cut(df_notnull[col], bins=cutOffPoints, retbins=True)
    woe_df, WOE_dict, iv, std = calcwoe(df_notnull, col_cut, 'tag')
    monotone = mono_or_u(woe_df['WOE'])
    whn = 0
    try:
        last_index = woe_df.shape[0] + 1
        woe_df.ix[last_index,col_cut] = np.nan
        woe_df.ix[last_index,'total'] = woe_df_null[1]
        woe_df.ix[last_index,'bad'] = woe_df_null[2]
        woe_df.ix[last_index,'good'] = woe_df_null[3]
        woe_df.ix[last_index,'good_pcnt'] = woe_df_null[4]
        woe_df.ix[last_index,'bad_pcnt'] = woe_df_null[5]
        woe_df.ix[last_index,'WOE'] = woe_df_null[6]
        woe_df.ix[last_index,'odds'] = woe_df_null[7]
        woe_df.ix[last_index,'total_pct'] = woe_df_null[8]
        woe_df.ix[last_index,'sub_iv'] = woe_df_null[9]

        iv = iv + iv_null+iv_sp
        woe_df['iv'] = iv
        df[col + '_cut'] = pd.cut(df[col], cutOffPoints)
        whn = 1
    except NameError:
        pass
    cut_cnt = len(woe_df)

    WOE_dict_new = {}
    for key in WOE_dict.keys():
            WOE_dict_new[key] = WOE_dict[key]['WOE']

    if whn == 1:
        WOE_dict_new[np.nan] = woe_df_null[6]
#改
        WOE_dict_new['-99999'] = woe_df_sp[6]
    df[col + '_woe'] = df[col + '_cut'].map(lambda x:WOE_dict_new[x]).astype(np.float64)
    return df[col + '_cut'], cutOffPoints, woe_df, [iv,  cut_cnt, monotone], df[col + '_woe'],WOE_dict_new







def calcwoe(df, col='', target='tag'):
    # 这里的tag里 好用户为0, 计算的woe是好坏比
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    bad.fillna(0,inplace=True)
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(inplace=True)
    regroup['good'] = regroup['total'] - regroup['bad']
    N = len(df)
    B = df[target].sum()
    G = N - B
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x*1.0/G)
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    # regroup['good_rate'] = regroup['good']/regroup['total']
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt) if x.bad_pcnt != 0 and x.good_pcnt != 0
    else 0, axis=1)
    regroup['odds'] = regroup.apply(lambda x:1.0*x.good/x.bad if x.bad !=0 else -1,axis=1)
    regroup['total_pct'] = regroup['total']/(1.0*df.shape[0])
    WOE_dict = regroup[[col,'total','bad','good','good_pcnt','bad_pcnt','WOE','odds','total_pct']].set_index(col).to_dict(orient='index')
    # 计算WOE方差
    std = regroup['WOE'].describe()['std']
    IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt)
    if x.bad_pcnt != 0 and x.good_pcnt != 0 else 0, axis=1)
    IV = sum(IV)
    regroup['sub_iv'] = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt)
    if x.bad_pcnt != 0 and x.good_pcnt != 0 else 0, axis=1)
    regroup['iv'] = IV
    return regroup, WOE_dict, IV, std





# 保存变量的分箱、cutpoint、woe、iv
def save_var_bins_woe_iv(df,cal,path,sub_path):
    df['cut_cnt'] = cal.T['cut_count']
    df['monotone'] = cal.T['monotone']
    df.rename(columns={'total':'internal_cnt','good':'internal_good_cnt','bad':'internal_bad_cnt','good_pcnt':'total_good_pct','bad_pcnt':'total_bad_pct'},inplace=True)

    df['total_good_pct'] = df['total_good_pct'].apply(lambda x : round(x,2))
    df['total_bad_pct'] = df['total_bad_pct'].apply(lambda x:round(x,2))

    col0 = df.columns[0]
    df[col0] = df[col0].map(lambda x:str(x))
    # df.to_csv(path + '/woe/%s_woe.csv' % var_col)
    df.to_csv(path + '/models/' +sub_path +'/%s_woe.csv' % col0,header=True)


def mono_or_u(s):      # 判断单调或U型
    s = s.reindex(s.values).index
    if s.is_monotonic:
        return 'increasing'
    elif s.is_monotonic_decreasing:
        return 'decreasing'
    elif s[0: s.argmax()].is_monotonic and s[s.argmax():].is_monotonic_decreasing:
        return 'upu'
    elif s[s.argmin():].is_monotonic and s[0: s.argmin()].is_monotonic_decreasing:
        return 'downu'
    else:
        return False





########################################################################
############# 变量精分箱
########################################################################


# 计算卡方
def Chi2(df, total_col, bad_col, overallRate):
    df2 = df.copy()
    df2['expected'] = df2[total_col].apply(lambda x: x*overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0]-i[1])**2/i[0] for i in combined]
    chi2 = sum(chi)
    # chi2= stats.chisquare(df[bad_col], df2['expected'])
    return chi2


# 基于最小卡方的最优分箱
def ChiMerge_MinChisq(df, col, target, confidenceVal=5, init=11):
#    print('the var for bins is %s' % col)
#    print('confidenceVal is %d' % confidenceVal)
#    print('init is %d' % init)
    if df[col].isnull().sum() > 0:
#        print(' ----- the var for bins include null')
        df1 = df[df[col].isnull()][[col, 'tag']]
        total = len(df1)
        bad = df1['tag'].sum()
        good = sum(df1['tag'] == 0)
        # 所有好用户中的占比
        bad_pcnt = 1.0 * bad / df['tag'].sum()
        # 所有坏用户中的占比
        good_pcnt = 1.0 * good / sum(df['tag'] == 0)
        # 区间内好坏用户比
        odds = 1.0*good/bad
        total_pct = round(1.0*total / df.shape[0],2)
        woe = np.log(good_pcnt * 1.0 / bad_pcnt) if good_pcnt != 0 and bad_pcnt !=0 else 0
#        iv_null = (good_pcnt-bad_pcnt)*np.log(good_pcnt*1.0/bad_pcnt)
        iv_null = (good_pcnt-bad_pcnt)*np.log(good_pcnt*1.0/bad_pcnt) if bad_pcnt!=0 else 0   #改
        woe_df_null = [np.nan, total, bad, good,good_pcnt, bad_pcnt, woe,odds,total_pct,iv_null]
        print ('have nan')
        df_notnull = df
    else:
        df_notnull = df
#改        
    if len(df[df[col] == -99999]) > 0:
#        print(' ----- the var for bins include null')
        df2 = df[(df[col] == -99999)][[col, 'tag']]
        total2 = len(df2)
        bad2 = df2['tag'].sum()
        good2 = sum(df2['tag'] == 0)
        # 所有好用户中的占比
        bad_pcnt2 = 1.0 * bad2 / df['tag'].sum()
        # 所有坏用户中的占比
        good_pcnt2 = 1.0 * good2 / sum(df['tag'] == 0)
        # 区间内好坏用户比
        odds2 = 1.0*good2/bad2
        total_pct2 = round(1.0*total2 / df.shape[0],2)
        woe2 = np.log(good_pcnt2 * 1.0 / bad_pcnt2) if good_pcnt2 != 0 and bad_pcnt2 !=0 else 0
#        iv_null = (good_pcnt-bad_pcnt)*np.log(good_pcnt*1.0/bad_pcnt)
        iv_sp = (good_pcnt2-bad_pcnt2)*np.log(good_pcnt2*1.0/bad_pcnt2) if bad_pcnt2!=0 else 0   #改
        woe_df_sp = [-99999, total2, bad2, good2,good_pcnt2, bad_pcnt2, woe2,odds2,total_pct2,iv_sp]
        print ('having -99999')
        df_notnull = df
    else:
        df_notnull = df
    colLevels = set(df_notnull[col])
    colLevels = sorted(list(colLevels))
    groupIntervals = [[i] for i in colLevels]
    if len(colLevels) > init:
        cutOffPoints = list(set(df_notnull[(df_notnull[col]!=-99999)][col].quantile(np.linspace(0.05,0.95,num=init))))
        cutOffPoints.append(-np.inf)
        cutOffPoints.append(np.inf)
        cutOffPoints.sort()
        print('cutOffPoints :')
        print(cutOffPoints)
        df_notnull[col+'_cut'],tmp2 = pd.cut(df_notnull[(df_notnull[col]!=-99999)][col], bins=cutOffPoints, retbins=True)
        colLevels = set(df_notnull[col+'_cut'])

        cutpoint = []
        for i in colLevels:
            reg = re.compile('-?[0-9]+[\.]?[0-9]*[e]?[\+]?[0-9]*')
            coll = list(map(float, re.findall(reg, str(i))))
            cutpoint.append(coll)
        groupIntervals = sorted(cutpoint)

    groupIntervals[0].insert(0,-np.inf)
    groupIntervals[-1].insert(0,np.inf)
    groupNum = len(groupIntervals)
    total = df_notnull.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad.fillna(0,inplace=True)
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    # 当前样本整体的坏样本率：坏样本数量/样本总量
    overallRate = B*1.0/N
    # print('overallRate is 0.2f') % overallRate
    while(1):
#        print('groupNum :')
#        print(groupIntervals)
#        print(len(groupIntervals))
        if len(groupIntervals) == 1:
            print ('nihao')
            cutOffPoints = groupIntervals[0]
            cutOffPoints.insert(0, -np.inf)
            cutOffPoints[-1] = np.inf
            cutOffPoints = list(set(cutOffPoints))
            cutOffPoints.sort()
            df_notnull[col + '_cut'] = pd.cut(df_notnull[(df_notnull[col]!=-99999)][col], cutOffPoints)
            woe_df, WOE_dict, iv, std = calcwoe(df_notnull, col + '_cut', 'tag')
            monotone = mono_or_u(woe_df['WOE'])
            whn = 0
            try:
                last_index = woe_df.shape[0] + 1
                woe_df.ix[last_index,col + '_cut'] = np.nan
                woe_df.ix[last_index,'total'] = woe_df_null[1]
                woe_df.ix[last_index,'bad'] = woe_df_null[2]
                woe_df.ix[last_index,'good'] = woe_df_null[3]
                woe_df.ix[last_index,'good_pcnt'] = woe_df_null[4]
                woe_df.ix[last_index,'bad_pcnt'] = woe_df_null[5]
                woe_df.ix[last_index,'WOE'] = woe_df_null[6]
                woe_df.ix[last_index,'odds'] = woe_df_null[7]
                woe_df.ix[last_index,'total_pct'] = woe_df_null[8]
                woe_df.ix[last_index,'sub_iv'] = woe_df_null[9]
                iv = iv + iv_null

            except UnboundLocalError:
                pass
            try:
                last_index2 = woe_df.shape[0]+1
                woe_df[col+'_cut'] = woe_df[col+'_cut'].astype('object')
                woe_df.ix[last_index2,col+'_cut'] = '-99999'
                woe_df.ix[last_index2,'total'] = woe_df_sp[1]
                woe_df.ix[last_index2,'bad'] = woe_df_sp[2]
                woe_df.ix[last_index2,'good'] = woe_df_sp[3]
                woe_df.ix[last_index2,'good_pcnt'] = woe_df_sp[4]
                woe_df.ix[last_index2,'bad_pcnt'] = woe_df_sp[5]
                woe_df.ix[last_index2,'WOE'] = woe_df_sp[6]
                woe_df.ix[last_index2,'odds'] = woe_df_sp[7]
                woe_df.ix[last_index2,'total_pct'] = woe_df_sp[8]
                woe_df.ix[last_index2,'sub_iv'] = woe_df_sp[9]
                iv = iv + iv_sp
            except UnboundLocalError:
                pass
            #iv = iv + iv_sp
            woe_df['iv'] = iv
            df[col + '_cut'] = pd.cut(df[(df[col]!=-99999)][col], cutOffPoints)
            whn = 1        

            cut_cnt = len(woe_df)
            WOE_dict_new = {}
            for key in WOE_dict.keys():
                WOE_dict_new[key] = WOE_dict[key]['WOE']
            if whn == 1:
                WOE_dict_new[np.nan] = woe_df_null[6]
#改                
                WOE_dict_new['-99999'] = woe_df_sp[6]
            df[col + '_woe'] = df[col + '_cut'].map(lambda x:WOE_dict_new[x]).astype(np.float64)
            return df[col + '_cut'], cutOffPoints, woe_df, [iv,  cut_cnt, monotone],df[col + '_woe'],WOE_dict_new

        elif len(groupIntervals) == 2:
            print ('nibuhao')
            groupIntervals = [sorted(i) for i in groupIntervals]
            cutOffPoints = [i[-1] for i in groupIntervals]
            # return cutOffPoints
            cutOffPoints.insert(0, -np.inf)
            cutOffPoints[-1] = np.inf
            cutOffPoints = list(set(cutOffPoints))
            cutOffPoints.sort()
            df_notnull[col + '_cut'] = pd.cut(df_notnull[col], cutOffPoints)
            woe_df, WOE_dict, iv, std = calcwoe(df_notnull, col + '_cut', 'tag')
            monotone = mono_or_u(woe_df['WOE'])
#            print (woe_df_null)
            try:
                woe_df_null = woe_df_null
                last_index = woe_df.shape[0] + 1
                woe_df.ix[last_index,col + '_cut'] = np.nan
                woe_df.ix[last_index,'total'] = woe_df_null[1]
                woe_df.ix[last_index,'bad'] = woe_df_null[2]
                woe_df.ix[last_index,'good'] = woe_df_null[3]
                woe_df.ix[last_index,'good_pcnt'] = woe_df_null[4]
                woe_df.ix[last_index,'bad_pcnt'] = woe_df_null[5]
                woe_df.ix[last_index,'WOE'] = woe_df_null[6]
                woe_df.ix[last_index,'odds'] = woe_df_null[7]
                woe_df.ix[last_index,'total_pct'] = woe_df_null[8]
                woe_df.ix[last_index,'sub_iv'] = woe_df_null[9]
                iv = iv + iv_null

            except UnboundLocalError:
                pass
            try:
                woe_df_sp = woe_df_sp
                last_index2 = woe_df.shape[0]+1
                woe_df[col+'_cut'] = woe_df[col+'_cut'].astype('object')
                woe_df.ix[last_index2,col+'_cut'] = '-99999'
                woe_df.ix[last_index2,'total'] = woe_df_sp[1]
                woe_df.ix[last_index2,'bad'] = woe_df_sp[2]
                woe_df.ix[last_index2,'good'] = woe_df_sp[3]
                woe_df.ix[last_index2,'good_pcnt'] = woe_df_sp[4]
                woe_df.ix[last_index2,'bad_pcnt'] = woe_df_sp[5]
                woe_df.ix[last_index2,'WOE'] = woe_df_sp[6]
                woe_df.ix[last_index2,'odds'] = woe_df_sp[7]
                woe_df.ix[last_index2,'total_pct'] = woe_df_sp[8]
                woe_df.ix[last_index2,'sub_iv'] = woe_df_sp[9]
                iv = iv + iv_sp
            except UnboundLocalError:
                pass
            #iv = iv + iv_sp
            woe_df['iv'] = iv
            df[col + '_cut'] = pd.cut(df[(df[col]!=-99999)][col], cutOffPoints)
            whn = 1        

            cut_cnt = len(woe_df)
            WOE_dict_new = {}
            for key in WOE_dict.keys():
                WOE_dict_new[key] = WOE_dict[key]['WOE']
            if whn == 1:
                try:
                    WOE_dict_new[np.nan] = woe_df_null[6]
                except NameError:
                    pass
                try:
                    WOE_dict_new['-99999'] = woe_df_sp[6]
                except NameError:
                    pass
            df[col + '_woe'] = df[col + '_cut'].map(lambda x:WOE_dict_new[x]).astype(np.float64)
            return df[col + '_cut'], cutOffPoints, woe_df, [iv,  cut_cnt, monotone],df[col + '_woe'],WOE_dict_new

        chisqList = []
        # 计算每一个区间内的卡方
        for interval in groupIntervals:
#            print(interval)
            df2 = regroup[(regroup[col] >= min(interval)) & (regroup[col] < max(interval))]
            chisq = Chi2(df2, 'total', 'bad', overallRate)
            chisqList.append(chisq)
#            print(chisq)

        # 对卡方做归一化
        chisqList = list(map(lambda x: x/(len(groupIntervals)-1), chisqList))
        min_position = chisqList.index(min(chisqList))
        if min(chisqList) >= confidenceVal:
            break
        if min_position == 0:
            combinedPosition = 1
        elif min_position == groupNum - 1:
            combinedPosition = min_position - 1
        else:
            if chisqList[min_position - 1] <= chisqList[min_position + 1]:
                combinedPosition = min_position - 1
            else:
                combinedPosition = min_position + 1
        groupIntervals[min_position] = groupIntervals[min_position]+groupIntervals[combinedPosition]
        groupIntervals.remove(groupIntervals[combinedPosition])
        groupNum = len(groupIntervals)
    groupIntervals = [sorted(i) for i in groupIntervals]  # 对每组的数据从小到大排序
    cutOffPoints = [i[-1] for i in groupIntervals]  # 提取出每组的最大值，也就是分割点
    ###################################################
    cutOffPoints.insert(0, -np.inf)
    cutOffPoints[-1] = np.inf
    cutOffPoints = list(set(cutOffPoints))
    cutOffPoints.sort()
    df_notnull[col + '_cut'] = pd.cut(df_notnull[(df_notnull[col]!=-99999)][col], cutOffPoints)
    woe_df, WOE_dict, iv, std = calcwoe(df_notnull, col + '_cut', 'tag')
    monotone = mono_or_u(woe_df['WOE'])
    whn = 0
    try:
        woe_df_null = woe_df_null
        last_index = woe_df.shape[0] + 1
        woe_df.ix[last_index,col + '_cut'] = np.nan
        woe_df.ix[last_index,'total'] = woe_df_null[1]
        woe_df.ix[last_index,'bad'] = woe_df_null[2]
        woe_df.ix[last_index,'good'] = woe_df_null[3]
        woe_df.ix[last_index,'good_pcnt'] = woe_df_null[4]
        woe_df.ix[last_index,'bad_pcnt'] = woe_df_null[5]
        woe_df.ix[last_index,'WOE'] = woe_df_null[6]
        woe_df.ix[last_index,'odds'] = woe_df_null[7]
        woe_df.ix[last_index,'total_pct'] = woe_df_null[8]
        woe_df.ix[last_index,'sub_iv'] = woe_df_null[9]
        iv = iv + iv_null

    except UnboundLocalError:
        pass
    try:
        woe_df_sp = woe_df_sp
        last_index2 = woe_df.shape[0]+1
        woe_df[col+'_cut'] = woe_df[col+'_cut'].astype('object')
        woe_df.ix[last_index2,col+'_cut'] = '-99999'
        woe_df.ix[last_index2,'total'] = woe_df_sp[1]
        woe_df.ix[last_index2,'bad'] = woe_df_sp[2]
        woe_df.ix[last_index2,'good'] = woe_df_sp[3]
        woe_df.ix[last_index2,'good_pcnt'] = woe_df_sp[4]
        woe_df.ix[last_index2,'bad_pcnt'] = woe_df_sp[5]
        woe_df.ix[last_index2,'WOE'] = woe_df_sp[6]
        woe_df.ix[last_index2,'odds'] = woe_df_sp[7]
        woe_df.ix[last_index2,'total_pct'] = woe_df_sp[8]
        woe_df.ix[last_index2,'sub_iv'] = woe_df_sp[9]
        iv = iv + iv_sp
    except UnboundLocalError:
        pass
            #iv = iv + iv_sp
    woe_df['iv'] = iv
    df[col + '_cut'] = pd.cut(df[(df[col]!=-99999)][col], cutOffPoints)
    whn = 1        

    cut_cnt = len(woe_df)
    WOE_dict_new = {}
    for key in WOE_dict.keys():
        WOE_dict_new[key] = WOE_dict[key]['WOE']
    if whn == 1:
        try:
            WOE_dict_new[np.nan] = woe_df_null[6]
        except NameError:
            pass
        try:    
            WOE_dict_new['-99999'] = woe_df_sp[6]
        except NameError:
            pass
#        print('='*80)
#        print(WOE_dict_new)
    df[col + '_woe'] = df[col + '_cut'].map(lambda x:WOE_dict_new[x]).astype(np.float64)
    if monotone:
#        print(monotone)
        return df[col + '_cut'], cutOffPoints, woe_df, [iv,  cut_cnt, monotone],df[col + '_woe'],WOE_dict_new
    else:
        return ChiMerge_MinChisq(df, col, target, confidenceVal+10, init)




def ChiMerge_MinChisq_MaxInterval(df, col, target, confidenceVal=5,init=11,max_interval = 10):
#    print('the var for bins is %s' % col)
#    print('confidenceVal is %d' % confidenceVal)
#    print('init is %d' % init)
    if df[col].isnull().sum() > 0:
#        print(' ----- the var for bins include null')
        df1 = df[df[col].isnull()][[col, 'tag']]
        total = len(df1)
        bad = df1['tag'].sum()
        good = sum(df1['tag'] == 0)
        # 区间内坏用户占所有坏用户的比例
        bad_pcnt = 1.0 * bad / df['tag'].sum()
        # 区间内好用户占所有好用户的比例
        good_pcnt = 1.0 * good / sum(df['tag'] == 0)
        # good_rate = good / total
        odds = 1.0*good/bad
        total_pct = 1.0*total/df.shape[0]
        woe = np.log(good_pcnt * 1.0 / bad_pcnt) if good_pcnt != 0 and bad_pcnt !=0 else 0
#        iv_null = (good_pcnt-bad_pcnt)*np.log(good_pcnt*1.0/bad_pcnt)
        iv_null = (good_pcnt-bad_pcnt)*np.log(good_pcnt*1.0/bad_pcnt) if bad_pcnt!=0 else 0   #改
        woe_df_null = [np.nan, total, bad, good,good_pcnt, bad_pcnt, woe,odds,total_pct,iv_null]
        
        # print woe_df_null
        # df_notnull = df[df[col].notnull()]
        # 缺失值的类型会在pd.cut的时候自动过滤掉
        df_notnull = df
#改
    elif (df[col]==-99999).sum() > 0:
#        print(' ----- the var for bins include null')
        df2 = df[(df[col]==-99999)][[col, 'tag']]
        total2 = len(df2)
        bad2 = df2['tag'].sum()
        good2 = sum(df2['tag'] == 0)
        # 区间内坏用户占所有坏用户的比例
        bad_pcnt2 = 1.0 * bad2 / df['tag'].sum()
        # 区间内好用户占所有好用户的比例
        good_pcnt2 = 1.0 * good2 / sum(df['tag'] == 0)
        # good_rate = good / total
        odds2 = 1.0*good2/bad2
        total_pct2 = 1.0*total2/df.shape[0]
        woe2 = np.log(good_pcnt2 * 1.0 / bad_pcnt2) if good_pcnt2 != 0 and bad_pcnt2 !=0 else 0
#        iv_null = (good_pcnt-bad_pcnt)*np.log(good_pcnt*1.0/bad_pcnt)
        iv_sp = (good_pcnt2-bad_pcnt2)*np.log(good_pcnt2*1.0/bad_pcnt2) if bad_pcnt2!=0 else 0   #改
        woe_df_sp = [-99999, total2, bad2, good2,good_pcnt2, bad_pcnt2, woe2,odds2,total_pct2,iv_sp]
        # print woe_df_null
        # df_notnull = df[df[col].notnull()]
        # 缺失值的类型会在pd.cut的时候自动过滤掉
        df_notnull = df
        
        
    else:
        df_notnull = df
    colLevels = set(df_notnull[col])
    colLevels = sorted(list(colLevels))
    # print(colLevels)
    groupIntervals = [[i] for i in colLevels]
    # print(groupIntervals)
    if len(colLevels) > init:
        cutOffPoints = list(set(df_notnull[(df_notnull[col]!=-99999)][col].quantile(np.linspace(0.05,0.95,num=init))))
        cutOffPoints.append(-np.inf)
        cutOffPoints.append(np.inf)
        cutOffPoints.sort()
#        print('cutOffPoints :')
#        print(cutOffPoints)
        df_notnull[col+'_cut'],tmp2 = pd.cut(df_notnull[(df_notnull[col]!=-99999)][col], bins=cutOffPoints, retbins=True)
        colLevels = set(df_notnull[col+'_cut'])
        # colLevels = sorted(list(colLevels))
#        print(colLevels)
        cutpoint = []
        for i in colLevels:
            # print i
            reg = re.compile('-?[0-9]+[\.]?[0-9]*[e]?[\+]?[0-9]*')
            coll = list(map(float, re.findall(reg, str(i))))
            cutpoint.append(coll)
        groupIntervals = sorted(cutpoint)

    groupIntervals[0].insert(0,-np.inf)
    groupIntervals[-1].insert(0,np.inf)
    groupNum = len(groupIntervals)
    total = df_notnull.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df_notnull.groupby([col])[target].sum()
    bad.fillna(0,inplace=True)
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B*1.0/N
    # print('overallRate is 0.2f') % overallRate
    while(1):
#        print('groupNum :')
#        print(groupIntervals)
#        print(len(groupIntervals))
        if len(groupIntervals) == 1:
            # return groupIntervals[0]
            cutOffPoints = groupIntervals[0]
            cutOffPoints.insert(0, -np.inf)
            cutOffPoints[-1] = np.inf
            cutOffPoints = list(set(cutOffPoints))
            cutOffPoints.sort()
            df_notnull[col + '_cut'] = pd.cut(df_notnull[(df_notnull[col]!=-99999)][col], cutOffPoints)
            woe_df, WOE_dict, iv, std = calcwoe(df_notnull, col + '_cut', 'tag')
            monotone = mono_or_u(woe_df['WOE'])
            whn = 0
            try:
                print(woe_df_null)
                last_index = woe_df.shape[0] + 1
                woe_df.ix[last_index,col + '_cut'] = np.nan
                woe_df.ix[last_index,'total'] = woe_df_null[1]
                woe_df.ix[last_index,'bad'] = woe_df_null[2]
                woe_df.ix[last_index,'good'] = woe_df_null[3]
                woe_df.ix[last_index,'good_pcnt'] = woe_df_null[4]
                woe_df.ix[last_index,'bad_pcnt'] = woe_df_null[5]
                woe_df.ix[last_index,'WOE'] = woe_df_null[6]
                woe_df.ix[last_index,'odds'] = woe_df_null[7]
                woe_df.ix[last_index,'total_pct'] = woe_df_null[8]
                woe_df.ix[last_index,'sub_iv'] = woe_df_null[9]

                # WOE_dict[np.nan] = woe_df_null[6]
                iv = iv + iv_null+iv_sp
                woe_df['iv'] = iv
                df[col + '_cut'] = pd.cut(df[(df[col]!=-99999)][col], cutOffPoints)
                whn = 1

            except: #NameError:
                pass
            cut_cnt = len(woe_df)
            WOE_dict_new = {}
            for key in WOE_dict.keys():
                WOE_dict_new[key] = WOE_dict[key]['WOE']
            if whn == 1:
                WOE_dict_new[np.nan] = woe_df_null[6]
#改                
                WOE_dict_new['-99999'] = woe_df_sp[6]
            df[col + '_woe'] = df[col + '_cut'].map(lambda x:WOE_dict_new[x]).astype(np.float64)
            return df[col + '_cut'], cutOffPoints, woe_df, [iv,  cut_cnt, monotone],df[col + '_woe'],WOE_dict_new

        if len(groupIntervals) == 2:
            groupIntervals = [sorted(i) for i in groupIntervals]
            cutOffPoints = [i[-1] for i in groupIntervals]
            # return cutOffPoints
            cutOffPoints.insert(0, -np.inf)
            cutOffPoints[-1] = np.inf
            cutOffPoints = list(set(cutOffPoints))
            cutOffPoints.sort()
            df_notnull[col + '_cut'] = pd.cut(df_notnull[(df_notnull[col]!=-99999)][col], cutOffPoints)
            woe_df, WOE_dict, iv, std = calcwoe(df_notnull, col + '_cut', 'tag')
            monotone = mono_or_u(woe_df['WOE'])
            whn = 0
            try:
#                print(woe_df_null)
                last_index = woe_df.shape[0] + 1
                woe_df.ix[last_index,col + '_cut'] = np.nan
                woe_df.ix[last_index,'total'] = woe_df_null[1]
                woe_df.ix[last_index,'bad'] = woe_df_null[2]
                woe_df.ix[last_index,'good'] = woe_df_null[3]
                woe_df.ix[last_index,'good_pcnt'] = woe_df_null[4]
                woe_df.ix[last_index,'bad_pcnt'] = woe_df_null[5]
                woe_df.ix[last_index,'WOE'] = woe_df_null[6]
                woe_df.ix[last_index,'odds'] = woe_df_null[7]
                woe_df.ix[last_index,'total_pct'] = woe_df_null[8]
                woe_df.ix[last_index,'sub_iv'] = woe_df_null[9]

                # WOE_dict[np.nan] = woe_df_null[6]
                iv = iv + iv_null+iv_sp
                woe_df['iv'] = iv
                df[col + '_cut'] = pd.cut(df[(df[col]!=-99999)][col], cutOffPoints)
                whn = 1

            except:# NameError:
                pass
            cut_cnt = len(woe_df)
            WOE_dict_new = {}
            for key in WOE_dict.keys():
                WOE_dict_new[key] = WOE_dict[key]['WOE']
            if whn == 1:
                WOE_dict_new[np.nan] = woe_df_null[6]
#改                
                WOE_dict_new['-99999'] = woe_df_sp[6]              
            df[col + '_woe'] = df[col + '_cut'].map(lambda x:WOE_dict_new[x]).astype(np.float64)
            return df[col + '_cut'], cutOffPoints, woe_df, [iv,  cut_cnt, monotone],df[col + '_woe'],WOE_dict_new

        chisqList = []
        for interval in groupIntervals:
            df2 = regroup[(regroup[col] >= min(interval)) & (regroup[col] < max(interval))]
            chisq = Chi2(df2, 'total', 'bad', overallRate)
            chisqList.append(chisq)
        chisqList = list(map(lambda x: x/(len(groupIntervals)-1), chisqList))
        min_position = chisqList.index(min(chisqList))
        if (min(chisqList) >= confidenceVal) or (groupNum <= max_interval):
            break
        if min_position == 0:
            combinedPosition = 1
        elif min_position == groupNum - 1:
            combinedPosition = min_position - 1
        else:
            if chisqList[min_position - 1] <= chisqList[min_position + 1]:
                combinedPosition = min_position - 1
            else:
                combinedPosition = min_position + 1
        groupIntervals[min_position] = groupIntervals[min_position]+groupIntervals[combinedPosition]
        groupIntervals.remove(groupIntervals[combinedPosition])
        groupNum = len(groupIntervals)
    groupIntervals = [sorted(i) for i in groupIntervals]  # 对每组的数据从小到大排序
    cutOffPoints = [i[-1] for i in groupIntervals]  # 提取出每组的最大值，也就是分割点
    ###################################################
    cutOffPoints.insert(0, -np.inf)
    cutOffPoints[-1] = np.inf
    cutOffPoints = list(set(cutOffPoints))
    cutOffPoints.sort()
    df_notnull[col + '_cut'] = pd.cut(df_notnull[(df_notnull[col]!=-99999)][col], cutOffPoints)
    woe_df, WOE_dict, iv, std = calcwoe(df_notnull, col + '_cut', 'tag')
    monotone = mono_or_u(woe_df['WOE'])
    groupNum = len(groupIntervals)
    whn = 0
    try:
#        print(woe_df_null)
        last_index = woe_df.shape[0] + 1
        woe_df.ix[last_index,col + '_cut'] = np.nan
        woe_df.ix[last_index,'total'] = woe_df_null[1]
        woe_df.ix[last_index,'bad'] = woe_df_null[2]
        woe_df.ix[last_index,'good'] = woe_df_null[3]
        woe_df.ix[last_index,'good_pcnt'] = woe_df_null[4]
        woe_df.ix[last_index,'bad_pcnt'] = woe_df_null[5]
        woe_df.ix[last_index,'WOE'] = woe_df_null[6]
        woe_df.ix[last_index,'odds'] = woe_df_null[7]
        woe_df.ix[last_index,'total_pct'] = woe_df_null[8]
        woe_df.ix[last_index,'sub_iv'] = woe_df_null[9]

        # WOE_dict[np.nan] = woe_df_null[6]
        iv = iv + iv_null+iv_sp
        woe_df['iv'] = iv
        df[col + '_cut'] = pd.cut(df[(df[col]!=-99999)][col], cutOffPoints)
        whn = 1

    except:# NameError:
        pass
    cut_cnt = len(woe_df)
    WOE_dict_new = {}
    for key in WOE_dict.keys():
        WOE_dict_new[key] = WOE_dict[key]['WOE']
    if whn == 1:
#        print(woe_df_null)
        WOE_dict_new[np.nan] = woe_df_null[6]
#改   
        WOE_dict_new['-99999'] = woe_df_sp[6]
    df[col + '_woe'] = df[col + '_cut'].map(lambda x:WOE_dict_new[x]).astype(np.float64)
    if monotone or groupNum <= max_interval:
#        print(monotone)
        return df[col + '_cut'], cutOffPoints, woe_df, [iv,  cut_cnt, monotone],df[col + '_woe'],WOE_dict_new
    else:
        return ChiMerge_MinChisq_MaxInterval(df, col, target, confidenceVal+10, init,max_interval = 10)




# 采用最小方差作为停止迭代标准
def vars_careful_woe_iv(data, feature, confidenceVal=1,sub_path = 'careful_woe',init=30,path='./',ifsave=False):
    cal = pd.DataFrame(index=['iv', 'cut_count', 'monotone'])
    woe_dict = {}
    for i in feature:
        print(i)
        if data[i].dtype == object:
            data[i + '_cut'] = data[i]
            woe_df, _, iv, std = calcwoe(data, i + '_cut')
            mono = mono_or_u(woe_df['WOE'])
#            print(woe_df['WOE'])
            cut_cnt = len(woe_df)
            iv_mono = [iv, cut_cnt, mono]
        else:
            data[i + '_cut'], cutpoint, woe_df, iv_mono,data[i + '_woe'],WOE_dict_new = ChiMerge_MinChisq(data, i, 'tag', confidenceVal, init)
            
            #if data[i + '_woe'].isnull().sum() > 0:\
            try:
                data.ix[data[i + '_woe'].isnull(),i + '_woe'] = WOE_dict_new[np.nan]
            #elif len(data[data[i] == -99999]) > 0:
            except KeyError:
                pass 
            try:
                data.ix[data[data[i] == -99999][i],i+'_woe'] = WOE_dict_new['-99999']
            except KeyError:
                pass
        # ks = (data, i + '_cut')
        # maxbin = woe_df['total'].max() / len(data)
        cal[i] = iv_mono
        try:
            woe_df['cut_point'] = [cutpoint]*len(woe_df)
        except:
            woe_df['cut_point'] = np.nan
        woe_dict[i] = woe_df[[i+'_cut','cut_point','total','good','bad','good_pcnt','bad_pcnt','WOE','odds','total_pct','sub_iv','iv']]
        if ifsave == True:
            save_var_bins_woe_iv(woe_dict[i],cal[i],path,sub_path)
    return woe_dict,cal



# 采用最小方差和最大分箱数作为停止迭代标准
# 需要增加判断string类型变量为null的情况 bug
def vars_careful_woe_iv_2(data, feature, confidenceVal=1,sub_path='careful_woe',init=30,path='./',ifsave=False):
    cal = pd.DataFrame(index=['iv', 'cut_count', 'monotone'])
    woe_dict = {}
    for i in feature:
        if data[i].dtype == object:
            data[i + '_cut'] = data[i]
            woe_df, _, iv, std = calcwoe(data, i + '_cut')
            mono = mono_or_u(woe_df['WOE'])
#            print(woe_df['WOE'])
            cut_cnt = len(woe_df)
            iv_mono = [iv, cut_cnt, mono]
            data[i + '_cut'].fillna('NaN',inplace=True)
            
            tmp = data[i + '_cut'].value_counts(dropna=False)
            tmp.sort_index(ascending=True,inplace=True)
            cutpoint = list(tmp.index)
            cutpoint = [x for x in cutpoint if x != 'NaN']
            data[i + '_cut'].replace('NaN',np.nan,inplace=True)
        else:
            data[i + '_cut'], cutpoint, woe_df, iv_mono,data[i + '_woe'],WOE_dict_new = ChiMerge_MinChisq_MaxInterval(data, i, 'tag', confidenceVal, init)
#            print(WOE_dict_new)
            if data[i + '_woe'].isnull().sum() > 0:
                data.ix[data[i + '_woe'].isnull(),i + '_woe'] = WOE_dict_new[np.nan]
#改
            if (data[i + '_woe']==-99999).sum() > 0:
                data.ix[(data[i + '_woe']==-99999),i + '_woe'] = WOE_dict_new['-99999']
        # k = ks(data, i + '_cut')
        # maxbin = woe_df['total'].max() / len(data)
        cal[i] = iv_mono
        try:
            woe_df['cut_point'] = [cutpoint]*len(woe_df)
        except:
            woe_df['cut_point'] = np.nan
        woe_dict[i] = woe_df[[i+'_cut','cut_point','total','good','bad','good_pcnt','bad_pcnt','WOE','odds','total_pct','sub_iv','iv']]
        if ifsave == True:
            save_var_bins_woe_iv(woe_dict[i],cal[i],path,sub_path)
    return woe_dict,cal





########################################################################
############# 计算变量各类情况下的psi
########################################################################


# 等深等宽方式计算psi
def model_evaluate_psi(df_train,col,df_test,cut_type='cut'):
	""" 违约概率十等分 - 等宽 或者 等深
	"""
	expect_data = df_train[col].fillna(-9999999)
	# print(expect_data.value_counts())
	if cut_type == 'cut':
		bins = np.arange(0,1.1,0.1)
		bins.insert(0,-np.inf)
		bins.insert(-1,np.inf)
		bins.sort()
#		print(u'正在等宽分割')
		expect_data['bins'] = pd.cut(expect_data,bins=bins)
	else :
#		print(u'正在等深分割')
		bins = np.percentile(expect_data.values,np.arange(0,110,10))
		bins = list(set(bins))
		bins.insert(0,-np.inf)
		bins.insert(-1,np.inf)
		bins.sort()
#		print('bins =  {0}'.format(bins))
		expect = pd.cut(expect_data,bins=bins)
		expect_data = pd.DataFrame(expect)

	expect_data_df_group = expect_data.groupby(by = [col])[col].size().fillna(0)
	expect_data_df_group = pd.DataFrame(expect_data_df_group)
	expect_data_df_group.rename(columns={col : 'interval_cnt'},inplace=True)
	expect_data_df_group['interval_cnt_rate'] = expect_data_df_group['interval_cnt']/expect_data_df_group['interval_cnt'].sum()
	expect_data_df_group.loc[:,'interval_cnt_rate'] = expect_data_df_group['interval_cnt_rate'].apply(lambda x : round(x,4))
	expect_data_df_group.reset_index(inplace=True)

	actual_data = df_test[col].fillna(-9999999)
	if cut_type == 'cut':
		actual_data['bins'] = pd.cut(actual_data,bins=bins)
	else:
		actual = pd.cut(actual_data,bins=bins)
		actual_data = pd.DataFrame(actual)
	actual_data_df_group = actual_data.groupby(by=[col])[col].size().fillna(0)
	actual_data_df_group = pd.DataFrame(actual_data_df_group)
	actual_data_df_group.rename(columns={col : 'interval_cnt'},inplace=True)
	actual_data_df_group['interval_cnt_rate'] = actual_data_df_group['interval_cnt']/actual_data_df_group['interval_cnt'].sum()
	actual_data_df_group.loc[:,'interval_cnt_rate'] = actual_data_df_group['interval_cnt_rate'].apply(lambda x : round(x,4))
	actual_data_df_group.reset_index(inplace=True)



	result = pd.merge(actual_data_df_group,expect_data_df_group,on=col,how='left',suffixes=('_actual','_expect'))
	result['sub_psi'] = (result['interval_cnt_rate_actual'] - result['interval_cnt_rate_expect']).T*(np.log(result['interval_cnt_rate_actual']/result['interval_cnt_rate_expect']))
	# print('='*40)
	# print(result)
	# print('='*40)
	result['psi'] = result[result['sub_psi'] < np.inf]['sub_psi'].sum() 
	psi = round(result[result['sub_psi'] < np.inf]['sub_psi'].sum(),4) 
#	print('The PSI of Var : {0}  =  {1}'.format(col,psi))
	return result,psi





# 使用精分的分箱方式计算 变量psi
def cal_psi(df_train,df_test,col,print_psi_detail=False):
    expect_data = df_train[[col]]
#    print(expect_data[col].value_counts(dropna=False))
    expect_data_df_group = expect_data.groupby(by = [col])[col].size().fillna(0)
    expect_data_df_group = pd.DataFrame(expect_data_df_group)
    expect_data_df_group.rename(columns={col : 'interval_cnt'},inplace=True)
    expect_data_df_group['interval_cnt_rate'] = expect_data_df_group['interval_cnt']/expect_data_df_group['interval_cnt'].sum()
    expect_data_df_group.loc[:,'interval_cnt_rate'] = expect_data_df_group['interval_cnt_rate'].apply(lambda x : round(x,4))
    expect_data_df_group.reset_index(inplace=True)
    # print('='*40)
    # print(expect_data_df_group)

    actual_data = df_test[[col]]
    actual_data_df_group = actual_data.groupby(by=[col])[col].size().fillna(0)
    actual_data_df_group = pd.DataFrame(actual_data_df_group)
    actual_data_df_group.rename(columns={col : 'interval_cnt'},inplace=True)
    actual_data_df_group['interval_cnt_rate'] = actual_data_df_group['interval_cnt']/actual_data_df_group['interval_cnt'].sum()
    actual_data_df_group.loc[:,'interval_cnt_rate'] = actual_data_df_group['interval_cnt_rate'].apply(lambda x : round(x,4))
    actual_data_df_group.reset_index(inplace=True)
    # print('='*40)
    # print(actual_data_df_group)


    result = pd.merge(actual_data_df_group,expect_data_df_group,on=col,how='left',suffixes=('_actual','_expect'))
    result['sub_psi'] = (result['interval_cnt_rate_actual'] - result['interval_cnt_rate_expect']).T*(np.log(result['interval_cnt_rate_actual']/result['interval_cnt_rate_expect']))
    result['psi'] = round(result['sub_psi'].sum(),4)
    
    if print_psi_detail == True:
        print('='*40)
        print(result)
        print('='*40)
    df = pd.DataFrame([[col[:-4],round(result['sub_psi'].sum(),4)]],columns=['name','psi'])
    result['psi'] = result[result['sub_psi'] < np.inf]['sub_psi'].sum()
    psi = round(result[result['sub_psi'] < np.inf]['sub_psi'].sum(),4)
#    print('The PSI of Var : {0}  =  {1}'.format(col,psi))
    return result,df 

    
    
    
    
###########################################
# 利用粗分、精分的分箱，计算验证集、测试集的iv
       
    
def var_bins_map_woe_pkl(data,woe_dict,feature_selected):
    for col in feature_selected:
        print('the var for bins :{0}'.format(col))
        cutOffPoint = woe_dict[col]['cut_point'][0]
        print(cutOffPoint)
        woe_dict_new = {}
        woe_dict[col][col + '_cut'] = woe_dict[col][col + '_cut'].astype(np.str)
        tmp_dict = woe_dict[col][[col + '_cut','WOE']].to_dict(orient='index')
        for key in tmp_dict.keys():
            woe_dict_new[tmp_dict[key][col + '_cut']] = tmp_dict[key]['WOE']
            
        woe_dict_new_df = pd.DataFrame(woe_dict_new,index=[len(woe_dict_new)]).T
        if data[col].dtype == 'O':
            data[col + '_cut'] = data[col]
            data[col + '_cut'].fillna('NaN',inplace=True)
            data[col + '_cut'] = data[col + '_cut'].astype(np.str)
            woe_dict_new_ = woe_dict_new.copy()
            if 'nan' in woe_dict_new_.keys():
                woe_dict_new_['NaN'] = woe_dict_new_['nan']
            
        else:
            
            data[col + '_cut'],_ = pd.cut(data[col],bins=cutOffPoint,retbins=True)
            data[col + '_cut'] = data[col + '_cut'].astype(np.str)
            
        if data[col].dtype == 'O':
            data[col + '_woe'] = data[col + '_cut'].map(lambda x : woe_dict_new_[x])
            data[col + '_cut'].replace('NaN',np.nan,inplace=True)
            
        else:
            data[col + '_woe'] = data[col + '_cut'].map(lambda x : woe_dict_new[x]).astype(np.float64)
        if data[col + '_woe'].isnull().sum() > 0:
            data.ix[data[col + '_woe'].isnull(),col + '_woe'] = woe_dict_new_df[woe_dict_new_df.index.isnull()].values[0][0]
            
    return data 


def cal_vars_psi_use_careful_bins(df_train,df_test_set,woe_dict,cal,vars_in_model,print_psi_detail=False,if_train_bins=True,if_test_bins=True):
    if not if_train_bins:
        df_train = var_bins_map_woe_pkl(df_train,woe_dict,vars_in_model)
        
    if not if_test_bins:
        df_test_set_ = var_bins_map_woe_pkl(df_test_set,woe_dict,vars_in_model)
        
    cal = cal.T.sort_values(by='iv',ascending=False)
    cal_ = cal#[cal['iv'] >= 0.04]
    cal_index = list(cal_.index)
    print(cal_index)
    df_psi = pd.DataFrame(columns=['name','psi'])
    print(vars_in_model)
    psi_dict = {}
    for col in cal_index :
        print(col)
        if col in vars_in_model:
            print('='*40)
            print(df_test_set_[col + '_cut'].value_counts(dropna=False))
            result,tmp = cal_psi(df_train,df_test_set_,col + '_cut',print_psi_detail)
            psi_dict[col] = result
            print(tmp)
            df_psi = pd.concat([df_psi,tmp],axis=0)
            
    return psi_dict,df_psi 





########################################################################
############# 变量woe值相关性的变量筛选
########################################################################

def corr_select(var_IV_df,df,varByIV,roh_thresould = 0.9):
    var_IV_selected = {k: var_IV_df.ix[k,'iv'] for k in varByIV}
    var_IV_sorted = sorted(var_IV_selected.items(), key=lambda d: d[1], reverse=True)
    var_IV_sorted_cols = [i[0] for i in var_IV_sorted]
    removed_var = []
    # roh_thresould = 0.9
    for i in range(len(var_IV_sorted_cols) - 1):
        if var_IV_sorted_cols[i] not in removed_var:
            x1 = var_IV_sorted_cols[i] + "_woe"
            for j in range(i + 1, len(var_IV_sorted_cols)):
                if var_IV_sorted_cols[j] not in removed_var:
                    x2 = var_IV_sorted_cols[j] + "_woe"
                    roh = np.corrcoef([df[x1], df[x2]])[0, 1]
                    if abs(roh) >= roh_thresould:
#                        print('the correlation coeffient between {0} and {1} is {2}'.format(x1, x2, str(roh)))
                        if var_IV_df.ix[var_IV_sorted_cols[i],'iv'] > var_IV_df.ix[var_IV_sorted_cols[j],'iv']:
                            removed_var.append(var_IV_sorted_cols[j])
                        else:
                            removed_var.append(var_IV_sorted_cols[i])


    return removed_var




########################################################################
############# 多重共线性的变量筛选
########################################################################

from sklearn.linear_model import LinearRegression

def multi_colinearity_filter(df,var_IV_sortet_2):
    removed_var = []
    for i in range(len(var_IV_sortet_2)):
        x0 = df[var_IV_sortet_2[i] + '_woe']
        x0 = np.array(x0)
        X_Col = [k + '_woe' for k in var_IV_sortet_2 if k != var_IV_sortet_2[i]]
        X = df[X_Col]
        X = np.matrix(X)
        regr = LinearRegression()
        clr = regr.fit(X, x0)
        x_pred = clr.predict(X)
        R2 = 1 - ((x_pred - x0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()
        vif = 1 / (1 - R2)
#        print('the VIF of {0} is {1}'.format(var_IV_sortet_2[i], vif))
        if vif > 10:
            removed_var.append(var_IV_sortet_2[i])
#            print("Warning: the vif for {0} is {1}".format(var_IV_sortet_2[i], vif))

    return removed_var






########################################################################
############# 显著性的变量筛选
########################################################################
def var_significance(df,var_IV_sortet_3):

    var_WOE_list = [i + '_woe' for i in var_IV_sortet_3]
    y = df['tag']
    X = df[var_WOE_list]
    X['intercept'] = [1] * X.shape[0]
    LR = sm.Logit(y, X).fit()
    summary = LR.summary2()
    pvals = LR.pvalues
    pvals = pvals.to_dict()

    ### 删除不显著变量
    varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
    varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)
    while (len(varLargeP) > 0 and len(var_WOE_list) > 0):
        varMaxP = varLargeP[0][0]
        if varMaxP == 'intercept':
#            print('the intercept is not significant!')
            break
        var_WOE_list.remove(varMaxP)
        y = df['tag']
        X = df[var_WOE_list]
        X['intercept'] = [1] * X.shape[0]
        LR = sm.Logit(y, X).fit()
        summary = LR.summary2()
        pvals = LR.pvalues
        pvals = pvals.to_dict()
        varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
        varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)

    return var_WOE_list





########################################################################
############# 最大箱占比的变量筛选
########################################################################
def var_bin_pct(df,cols,pct_threshold = 0.9):
    remove_cols=[]
    for col in cols:
        tmp = df[col + '_cut'].value_counts(dropna=False,normalize=True)
        if (tmp > pct_threshold).any():
            remove_cols.append(col)

    return remove_cols






########################################################################
############# 利用粗分、精分的分箱，计算验证集、测试集变量的iv
########################################################################


def var_bins_map_woe_pkl(data,woe_dict,feature_selected):
    for col in feature_selected:
#        print('the var for bins :{0}'.format(col))
        cutOffPoint = woe_dict[col]['cut_point'][0]
#        print(cutOffPoint)
        woe_dict_new = {}
        woe_dict[col][col + '_cut'] = woe_dict[col][col + '_cut'].astype(np.str)
        tmp_dict = woe_dict[col][[col + '_cut','WOE']].to_dict(orient='index')
        for key in tmp_dict.keys():
            woe_dict_new[tmp_dict[key][col + '_cut']] = tmp_dict[key]['WOE']
            
        woe_dict_new_df = pd.DataFrame(woe_dict_new,index=[len(woe_dict_new)]).T
        if data[col].dtype == 'O':
            data[col + '_cut'] = data[col]
            data[col + '_cut'].fillna('NaN',inplace=True)
            data[col + '_cut'] = data[col + '_cut'].astype(np.str)
            woe_dict_new_ = woe_dict_new.copy()
            if 'nan' in woe_dict_new_.keys():
                woe_dict_new_['NaN'] = woe_dict_new_['nan']
            
        else:
            
            data[col + '_cut'],_ = pd.cut(data[col],bins=cutOffPoint,retbins=True)
            data[col + '_cut'] = data[col + '_cut'].astype(np.str)
            
        if data[col].dtype == 'O':
            data[col + '_woe'] = data[col + '_cut'].map(lambda x : woe_dict_new_[x])
            data[col + '_cut'].replace('NaN',np.nan,inplace=True)
            
        else:
            data[col + '_woe'] = data[col + '_cut'].map(lambda x : woe_dict_new[x]).astype(np.float64)
        if data[col + '_woe'].isnull().sum() > 0:
            data.ix[data[col + '_woe'].isnull(),col + '_woe'] = woe_dict_new_df[woe_dict_new_df.index.isnull()].values[0][0]
            
    return data 


   

########################################################################
############# LR模型训练及评估
########################################################################

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.externals import joblib

# 评价函数
# KS
def ks(y_predprob,y_train):
    from scipy.stats import ks_2samp
    tmp_pos = y_predprob[y_train == 1]
    tmp_neg = y_predprob[y_train == 0]
    res = ks_2samp(tmp_pos, tmp_neg)
    (KS,P) = ks_2samp(tmp_pos, tmp_neg)
#    print('KS',round(KS,4))
    # print(res)
    # KS = round(KS,4)
    return KS
    
    
# AUC
def auc(y_predprob,y_true):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_predprob, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
#    print('auc', round(roc_auc,4))
    return roc_auc
    
    
def model_params_optimization(X_train_1, X_test_1, y_train_1, y_test_1):
    model_parameter1 = {}
    model_parameter2 = {}
    for C_penalty in np.arange(0.01, 0.1, 0.01):
        for bad_weight in range(2, 11, 1):
            LR_model_2 = LogisticRegressionCV(Cs=[C_penalty], penalty='l1', solver='liblinear',
                                              class_weight={1: bad_weight, 0: 1},random_state=42)
            LR_model_2_fit = LR_model_2.fit(X_train_1, y_train_1)
            y_pred = LR_model_2_fit.predict_proba(X_test_1)[:, 1]
#            print('the model evaluate in valid set:')
            KS = ks(y_pred,y_test_1)
            roc_auc = auc(y_pred,y_test_1)

            model_parameter1[(C_penalty, bad_weight)] = (round(KS,4),round(roc_auc,4))
            model_parameter2[(C_penalty, bad_weight)] = round(KS,4)
#            print('the C_penalty is {0}, the bad_weight is {1}, the ks is {2},the auc is {3}'.format(C_penalty, bad_weight,round(KS,4),round(roc_auc,4)))
#            print('\n')

    return model_parameter1,model_parameter2











########################################################################
############# 从违约概率到评分的映射
########################################################################


def score_to_grade(score):

    """

tag 0.0 1.0 interval_bad_rate
score_bins
(349, 550]  4260.0  3684.0  0.463746
(550, 600]  9332.0  3206.0  0.255703
(600, 650]  12088.0 2229.0  0.155689
(650, 700]  7816.0  690.0   0.081119
(700, 800]  3177.0  113.0   0.034347
(800, 950]  50.0    0.0 0.000000

    """
    if (score >= 350) & (score < 551):
        return "Very_Bad"
    elif (score >= 551) & (score <601):
        return "Poor"
    elif (score >= 601) & (score <651):
        return "Fair"
    elif (score >= 651) & (score <701):
        return "Good"
    elif (score >= 701) & (score <801):
        return "Very_Good"
    elif (score >= 801) & (score <=950):
        return "Excellent"
    else:
        return "unknow"




def mx_scores(y_predprob):
    """
    y_predprob : 预测成为坏用户的概率【p】
    odds = 好/坏 = （1-p）/p

    odds 翻倍 评分增加 50

    700 = A + Blog(10)
    750 = A + Blog(20)
    
    上面基础上增加或减少50
    """
    df = pd.DataFrame(y_predprob,columns=['predprob'])
    y_predprob = df['predprob'].values
    scores = 533.91 + 72.13*np.log(np.abs((1- y_predprob))/y_predprob)
    for index in range(scores.shape[0]):
        if (scores[index] == np.inf) | (scores[index] == -np.inf):
            scores[index] = 0
        else:
            pass
    scores = np.array([round(sco) for sco in scores])
    scores = list(scores.astype(np.int))
    for index in range(len(scores)):
        if (scores[index] < 350) and scores[index] > 0:
            scores[index] = 350

        elif scores[index] > 950:
            scores[index] = 950

    df['score'] = scores
    df['grade'] = df['score'].apply(score_to_grade)

    
    return df
















