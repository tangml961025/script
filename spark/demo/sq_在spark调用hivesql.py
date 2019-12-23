import pandas as pd
import pymysql
from pyhive import hive
import pandas as pd
from pyspark import SparkContext,SQLContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
import datetime
import json
import urllib.request
import sys

dt1=datetime.datetime.strptime(sys.argv[1],'%Y-%m-%d')
dt=dt1.strftime('%Y-%m-%d')
YDAY=(dt1+datetime.timedelta(days=-1)).strftime('%Y-%m-%d')





spark=SparkSession \
        .builder \
        .config("spark.eventLog.enabled", "false") \
        .config("spark.dynamicAllocation.initialExecutors", "5") \
        .config("spark.dynamicAllocation.minExecutors", "2") \
        .config("spark.dynamicAllocation.maxExecutors", "10") \
        .config("spark.executor.memory", "4g")\
        .config("spark.driver.memory", "6g")\
        .config("spark.cores.max", "2")\
        .config("spark.task.maxFailures", "1000")\
        .config("spark.default.parallelism", "1000")\
        .config("spark.storage.memoryFraction","0.5")\
        .config("spark.shuffle.memoryFraction","0.3")\
        .config("spark.sql.shuffle.partitions","300")\
        .appName('dim_customer_career_all_clx') \
        .master('yarn')\
        .getOrCreate()

hql1 = """
insert overwrite table ods.ds_customer_career_all partition(bus_date='%s')
select t.user_id as user_id,
        t.company_name as company_name,
        t.company_address as company_address,
        t.company_phone as company_phone,
        max(t.data_source) as data_source,
        max(t.create_time) as create_time,
        current_timestamp() as update_time
from(
select  a.user_id as user_id,
        if(regexp_replace(b.company_name,'"','') is not null,regexp_replace(b.company_name,'"',''),'') as company_name,
        if(c.address_name is not null,c.address_name,'') as company_address,
        if(d.contact_number is not null, d.contact_number,'')as company_phone,
        '新核心' as data_source,
        a.create_time as create_time
from stg_loanuser.user_personal_basic_info a
left join ods.ds_customer_company_info b
on a.user_id=b.user_id
left join ods.ds_customer_address_info c
on a.user_id=c.user_id and  c.address_type='21'
left join ods.ds_customer_contact_info d   
on a.user_id=d.user_id and d.contact_type='220'
where a.dt='%s' and a.user_id is not null 
  and (b.company_name is not null or c.address_name is not null or d.contact_number is not null)
    and (c.update_time between concat('%s',' 00:00:00') and concat('%s',' 23:59:59')
    or d.update_time between concat('%s',' 00:00:00') and concat('%s',' 23:59:59'))
union 
select  a.user_id as user_id,
        if(c.employer is not null,c.employer,'') as company_name,
        if(c.employ_address is not null,c.employ_address,'') as company_address,
        if(c.office_telephone is not null,c.office_telephone,'')  as company_phone,
        '人行征信报告' as data_source,
        a.create_time as create_time
from 
(
select certno,employer,employ_address,office_telephone from renhang_user_profile.dim_customer_pbc_credit_report
where  dt='%s'
) c
inner join ods.ds_customer_cerificate_info a
on a.certificate_id=c.certno and a.invalid_status='1' and a.certificate_type='0'
where a.user_id is not null  
    and (c.employer is not null or c.employ_address is not null or c.office_telephone is not null)
)t
group by t.user_id,t.company_name,t.company_address,t.company_phone
"""%(dt,dt,dt,dt,dt,dt,dt)

spark.sql(hql1)
