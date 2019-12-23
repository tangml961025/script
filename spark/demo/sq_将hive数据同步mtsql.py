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


con=pymysql.connect(host="58.59.11.85",user="etl",
                   password="6sY2*(g3YTs131Kl",db="reportpublic",port=3306)
cur = con.cursor()
sql1 = "truncate table reportpublic.dim_customer_career_all"
pd=cur.execute(sql1)

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
        .appName('clx_mysql_dim_customer_career_all') \
        .master('yarn')\
        .getOrCreate()

table1 = spark.read \
    .format("jdbc") \
    .option("url","jdbc:mysql://58.59.11.85:3306") \
    .option("dbtable", "reportpublic.dim_customer_career_all") \
    .option("user", "etl") \
    .option("password", "6sY2*(g3YTs131Kl") \
    .load()
table1.createOrReplaceTempView("dim_customer_career_all")


spark.sql("""
insert into dim_customer_career_all
select
user_id,
trim(if(company_name is not null,company_name,'')) as name,
trim(if(company_address is not null,company_address,'')) as address,
trim(if(company_phone is not null,company_phone,'')) as phone,
max(data_source),
max(create_time),
max(update_time)
from ods.ds_customer_career_all
group by user_id,name,address,phone


""")
