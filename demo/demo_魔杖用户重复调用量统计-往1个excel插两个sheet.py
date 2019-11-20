#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import odps
with open('/Users/tangmeiling/Documents/1-魔蝎/ak/access.csv', 'r') as csv:
    access_csv = csv.readlines()
Id = access_csv[0].strip()
Key = access_csv[1]
o = odps.ODPS(Id,Key,'apiloganalysis')


# In[11]:


path = '/Users/tangmeiling/Desktop/'

tables = ['temp_tml_mz2_repeat_cust','temp_tml_mz1_repeat_cust']
i=1
writer = pd.ExcelWriter('{}魔杖重复值统计.xlsx'.format(path))
for tb in tables:
    data_odps = odps.DataFrame(o.get_table(tb))
    data_odps = data_odps.to_pandas()
    data_odps.rename(columns={'request_day':'时间','repeat_inquiry':'重复调用','distinct_repeat_cust':'人数','repeat_num':'调用总次数'},inplace=True)
    data_odps.to_excel(writer,sheet_name='{}'.format(i),index=False)
    i= i+1
writer.save()

