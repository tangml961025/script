#!/usr/bin/env python
# coding: utf-8

"""
以下是在odps调用127py脚本的语句
/data/anaconda3/bin/python3 路径/文件名.py
"""

import pandas as pd
import numpy as np
import os
import xlrd
import uuid

import odps
from odps import ODPS
import datetime
import time,datetime
from datetime import timedelta

import json
import logging
from logging.handlers import TimedRotatingFileHandler
import smtplib
from email.mime.text import MIMEText

# 过滤警告
import warnings
warnings.filterwarnings("ignore")


# 调用ak.py文件连接odps
from ak import get_odps_key
o=get_odps_key()



# ======================================================================================================
# 时间函数
# 可删除
# ======================================================================================================
def get_timezone():
    nowTime=datetime.datetime.now() #当天
    yestoday = nowTime+datetime.timedelta(days=-1) # 前一天
    timeformat = '%Y%m%d %H:%M:%S'
    timezone = nowTime.strftime(timeformat) #转化为字符串
    return timezone


# ======================================================================================================
# 获取odps数据，并处理为list的函数
# ======================================================================================================
def get_odps_data(sql,pt_plus1,n):
    bt_records=[]
    with o.execute_sql(sql.format(pt_plus1=pt_plus1),hints={"odps.sql.submit.mode" : "script"}).open_reader() as reader:
        bt_records = [[record[i] for i in range(0,n)] for record in reader]
    print(bt_records)
    print('='*60)
    return bt_records


# ======================================================================================================
# html
# 发送邮件函数
# ======================================================================================================
def send_mail(to_list,sub,get_pt,content1):  ########################### 修改⚠️
    me = ""+"<"+mail_user+"@"+mail_postfix+">"
    # 表的列名，⚠️修改
    table1_headers = ['source','dt','...']


    # 如果设置的时间没有数据，则发送邮件：there is no data today!
    # 如果没有这个需要可以删除
    if len(get_pt)==0:
        html = """
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <title>回溯备份自动化测试监控</title>
    </head>
    <body>
        <div id="container">
          <p><strong>监控：</strong></p>
          <p>监控统计时间: """ + timezone + """</p>
          <p>there is no data today!</p>
    </body>
</html>
"""

    # 如果有数据
    if (isinstance(get_pt,list)) & (len(get_pt)>0):
        # 邮件正文 标题
        html_header = """
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <title>回溯样本监控</title>
        <p>监控统计时间: """ + timezone + """</p>
    </head>
    <body>"""

        # 表格主体
        ############################ html1
        html1 = """
        <br />
        <p><strong>【PART1 样本监控】</strong></p>
        <div id="content" style="text-align:center">
            <table width="1000" border="2" bordercolor="black" cellspacing="0.5">
                <thead height="25" bgcolor="#d0d0d0">
                    <tr>
                    <strong>"""
        # 表头
        html1_headers_str=""""""
        for table_header in table1_headers:
            html1_headers_str+="""
                        <td >{}</td>""".format(table_header)
        # 表格内容       
        html1_table_values="""
                    </strong>
                    </tr>
                </thead>
                <tbody>"""
        for i in range(len(content1)):# 增加数据
            print('[1].样本监控',i,'='*10) # 可修改可删除⚠️
            h="""
                    <tr>"""
            for j in range(6): # 请根据你的数据列数修改⚠️
                h+="""
                        <td"""
                if j == 0 :
                    h+= """ align="center" width="160" bgcolor="#FFC125"  """ # 设置html中的表格的列格式

                h+=""">"""+str(content1[i][j])+"""</td>"""
            html1_table_values+=h+"""
                    </tr>"""
        html1_table_values+="""
                </tbody>
            </table>
        </div>"""
 
        
        #合并各部分html
        html = html_header+"""
        """+html1+html1_headers_str+html1_table_values+"""
    </body>
</html>"""



    msg = MIMEText(html, _subtype='html', _charset='utf8')#解决乱码
    msg['Subject'] = sub
    msg['From'] = me
    msg['To'] = ";".join(to_list)
    try:
        s = smtplib.SMTP_SSL(mail_host, port=465)
        s.login(mail_user+"@"+mail_postfix, mail_pass)
        s.sendmail(me, to_list, msg.as_string())
        s.close()
        print('sending email: OK')
    except Exception as e:
        # log.error('error in sending email:' + str(e))
        print('error in sending email:',str(e))




# ======================================================================================================
# sql ⚠️
# ======================================================================================================

sql_get_pt = """
select distinct source,dt,pt,sourcename
from modelgroup.athena_mz_all_backtest_per_day_recoreds_temp_tml;
"""

# 样本监控
sql_mz_all_yangben_check = """
select t.sourcename,t.dt,t1.encryption,t2.all_cnt,t2.all_md5_ratio
    ,case when t.mz2_inner != '-1' then concat('内部提供(',if(t.test_product regexp '魔杖2.0\\\(联合建模\\\)','联合建模|',''),if(t.test_product regexp '魔杖2.0\\\(内部提供所有\\\)','mz2+mz3|',''),if(t.test_product regexp '佩奇分','佩奇分|',''),if(t.test_product regexp '^宝莉分$','宝莉分|',''),if(t.test_product regexp '宝莉分增强版','宝莉分增强版|',''),if(t.test_product regexp '大象分','大象分|',''),if(t.test_product regexp '猛犸分','猛犸分|',''),')',';',if(t.mz2_outer regexp '魔杖',replace(t.mz2_outer,':0',''),''))
        else if(t.mz2_outer regexp '魔杖',replace(t.mz2_outer,':0',''),'') end as test_mz2_product
from modelgroup.athena_mz_all_backtest_per_day_recoreds_temp_tml t
left outer join (
    select source,dt
        ,case when record_name = 'md5' then concat(name_md5_if_gbk,';',record_idcard,';',record_mobile,' / ',length_name,';',length_idcard,';',length_mobile,'|',encryption_if_right)
            else concat(record_name,';',record_idcard,';',record_mobile,' / ',length_name,';',length_idcard,';',length_mobile,'|',encryption_if_right)
            end as encryption
        ,encryption_if_right
    from modelgroup.athena_mz_all_backtest_per_day_recoreds_temp_tml_encryption
)t1 on t.source = t1.source and t.dt = t1.dt
left outer join (
    select source,dt
        ,concat(upload_cnt,';',cnt,';',moxieid_cnt,'|',if(upload_cnt=cnt and cnt=moxieid_cnt,'1','0')) as all_cnt
        ,concat(round(100*md5_all_ratio,2),'%;',round(100*md5_name_ratio,2),'%;',round(100*md5_idcard_ratio,2),'%;',round(100*md5_mobile_ratio,2),'%','|',if(md5_all_ratio<0.05,'0',if(md5_all_ratio<0.5,'1','2'))) as all_md5_ratio
    from modelgroup.athena_mz_all_perday_md5_ratio_temp_tml
)t2 on t.source = t2.source and t.dt = t2.dt
order by t.dt asc limit 1000
;
"""


# ======================================================================================================
# 主体
# ======================================================================================================

# 获取odps数据，4、6请根据情况修改，意思是列数⚠️
get_pt=get_odps_data(sql_get_pt,'',4)
content1 = get_odps_data(sql_mz_all_yangben_check,'',6)


timezone = get_timezone() #获取监控时间
current = datetime.datetime.now().strftime('%Y%m%d')


# 填写邮箱
mail_host = "smtp.exmail.qq.com"
mail_user='tangmeiling' ##### 改:邮箱⚠️
mail_postfix='51dojo.com' ##### 改:邮箱⚠️
mail_pass='' ##### 改:密码⚠️
# 填写收件人
mailto_list=[
            'wanghuanan@51dojo.com'
            ]

sub="邮件标题"
send_mail(mailto_list,sub,get_pt,content1) #get_pt,content1两个参数可以自行修改⚠️
print('发送成功')