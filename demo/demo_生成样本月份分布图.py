
def draw_sample_distribution_map(a,mz3_type):
    sql_get_sample_distribution = '''
    SELECT sample_month,COUNT(sample_month) as month_num ,cast(SUM(has_mz_data) as int) as hit_num
    from(
        SELECT substring(request_time,1,7) as sample_month,has_mz_data
        from apiloganalysis.apollo_mz_all_feature_mz3_{0}_{1}
        -- 这里放魔杖3的结果表
    ) t
    GROUP by sample_month
    order by sample_month
    LIMIT 10000
    ;'''.format(mz3_type,a.source)
    with o.execute_sql(sql_get_sample_distribution).open_reader() as reader :
        result = reader.to_pandas()
        print(result)

    months = result['sample_month'].tolist()
    print(months)
    datas = np.array([result['month_num'].tolist(),result['hit_num'].tolist()]).T

    df = DataFrame(datas, columns=['sample_count','has_mz_data_count'],index = months)
    df.plot(kind=plt.bar, color=['darkblue','orange'])
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 20
    plt.xticks(rotation=60)
    plt.savefig('/home/sample_auto/mozhang_auto/monitor/mz3_sample_pics/'+a.sourcename+'.jpg')
    print('样本分布图已生成')