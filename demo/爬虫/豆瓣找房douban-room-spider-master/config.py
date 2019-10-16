# -*- coding: utf-8 -*-


# mail = {
#     'sender': 'xxx@163.com',
#     'host': 'smtp.163.com',
#     'receivers': ['xxx@qq.com'],
#     'password': 'password',
#     'subject_prefix': '豆瓣爬虫租房'
# }

# groups = [
#     (26926, '北京租房豆瓣'),
#     (279962, '北京租房（非中介）'),
#     (262626, '北京无中介租房（寻天使投资）'),
#     (35417, '北京租房'),
#     (56297, '北京个人租房 （真房源|无中介）'),
#     (257523, '北京租房房东联盟(中介勿扰) '),
# ]

# locations = ('西二旗', '安宁庄', '小米', '上地', '龙泽', '永泰庄', '清河')

# exclude_words = ('求租')



mail = {
    'sender': 'tangmeiling@51dojo.com',
    'host': 'smtp.exmail.qq.com',
    'receivers': ['tangmeiling@51dojo.com'],
    'password': 'imap.exmail.qq.com',
    'subject_prefix': '【租房】豆瓣爬虫'
}

groups = [
    (26926, '杭州租房（房东直租，中介勿进）'),
    (279962, '杭州城西租房'),
    (262626, '杭州西湖区租房'),
    (35417, '杭州无中介租房'),
    (56297, '杭州 出租 租房 中介免入'),
    (257523, '☀杭州租房大全【好评★★★★★】'),
]

locations = ('三坝', '文新', '丰潭路', '古翠路', '学院路', '金茂悦', '耀江文鼎苑','枫华府第','世纪新城')

exclude_words = ('求租','单间房','主卧','合租')
