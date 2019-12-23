import pandas as pd
# import pymysql
# import psycopg2
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
import os


# # python连接pgsql
# conn = psycopg2.connect(database='hyxf_prod',user='tangmeiling',password='tangmeiling123',host='10.10.2.8',port='3432')
# cur = conn.cursor()
# # cur.execute("")
# # rows = cur.fetchall()


##################################### 配置spark
###### spark连接hive，需要指定地址，调取相应jar包
# sparkClassPath = os.getenv('SPARK_CLASSPATH','/opt/cloudera/parcels/CDH-6.0.1-1.cdh6.0.1.p0.590678/lib/spark/jars/postgresql-42.2.8.jar')
spark=SparkSession \
        .builder \
        .config("spark.driver.extraClassPath", '/opt/cloudera/parcels/CDH-6.0.1-1.cdh6.0.1.p0.590678/lib/spark/jars/postgresql-42.2.8.jar') \
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
        .config("spark.driver.maxResultSize","4g")\
        .config("spark.kryoserializer.buffer.max","512m")\
        .config("spark.kryoserializer.buffer","64m")\
        .appName('tml_pgsql_temp_tml_test_20191125') \
        .master('yarn')\
        .getOrCreate()


##################################### 测试对hive数据库是否能正常操作
# 新建表
spark.sql("""
    create table dm.temp_tml_test_20191125_copy
    like dm.temp_tml_test_20191125
""")
# select
data_se1 = spark.sql("select * from dm.temp_tml_test_20191125")
data_se1.show()
# +----+---+--------+-----+----------+
# |name|age| carrier|score| date_time|
# +----+---+--------+-----+----------+
# | yyy| 16|liantong|   89|2019-11-24|
# | yyy| 16|liantong|   89|2019-11-24|
# | xxx| 23|  yidong|   93|2019-11-25|
# +----+---+--------+-----+----------+
# insert动态分区
data_insert1 = spark.sql("""
    set hive.exec.dynamic.partition = true;
    set hive.exec.dynamic.partition.mode = nonstrict;
    insert overwrite table dm.temp_tml_test_20191125_copy
    partition (date_time)
    select name,age,carrier,score,date_time from dm.temp_tml_test_20191125
""")
# insert静态分区
data_insert2 = spark.sql("""
    insert overwrite table dm.temp_tml_test_20191125_copy
    partition (date_time = '2019-09-11')
    values ('abc',null,'dianxin',40)
""")
# select
data_se2 = spark.sql("select * from dm.temp_tml_test_20191125_copy")
data2.show()
# +----+----+--------+-----+----------+
# |name| age| carrier|score| date_time|
# +----+----+--------+-----+----------+
# | abc|null| dianxin|   40|2019-09-11|
# | yyy|  16|liantong|   89|2019-11-24|
# | yyy|  16|liantong|   89|2019-11-24|
# | xxx|  23|  yidong|   93|2019-11-25|
# +----+----+--------+-----+----------+
############### 测试结束，OK.


##################################### 用jdbc连接postgresql
spark_table = spark.read \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://10.10.2.8:3432/hyxf_prod") \
        .option("dbtable", "dm.temp_tml_test_20191125") \
        .option("user", "tangmeiling") \
        .option("password", "tangmeiling123") \
        .load()

# spark生成一个视图
spark_table.createOrReplaceTempView("pgsql_temp_tml_test_20191125")

# 运行sql，将hive数据插入视图，即将数据插入pgsql表
spark.sql("""
    insert into table pgsql_temp_tml_test_20191125
    select *
    from dm.temp_tml_test_20191125;
""")



