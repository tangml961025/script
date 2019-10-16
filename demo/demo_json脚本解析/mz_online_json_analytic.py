import pandas as pd
import numpy as np
import json
import os

class print_json:
    def __init__(self):
        pass
    def print_keyvalue_all(self,input_json):
        key_value=''
        if isinstance(input_json,dict):
            for key in input_json.keys():
                key_value = input_json.get(key)
                if isinstance(key_value,dict):
                    self.print_keyvalue_all(key_value)
                elif isinstance(key_value,list):
                    for json_array in key_value:
                        self.print_keyvalue_all(json_array)
                else:
                    # print(str(key)+","+str(key_value))
                    tmp = pd.DataFrame([[key,key_value]],columns=['key','value'])
                    global df
                    df = pd.concat([df,tmp],axis=0)

        elif isinstance(input_json,list):
            for input_json_array in input_json:
                self.print_keyvalue_all(input_json_array)


df = pd.DataFrame(columns=['key','value'])
test = print_json()

curr_dir = os.path.realpath(__file__)
print(curr_dir[:-len('mz_online_json_analytic.py')])
dirs = os.listdir(np.str(curr_dir[:-len('mz_online_json_analytic.py')]))
print(dirs)

for file in dirs:
    if file != '.DS_Store' and file != 'result' and file != 'mz_online_json_analytic.py':
        file_name = file[:-5]
        print('file_name is %s' % file_name)
        with open(file) as f:
            input_json = json.load(f)

        input_json = input_json['data']['mobile_info']
        test.print_keyvalue_all(input_json)
        df.to_csv('./result/' + file_name + '.csv',index=False)
        df = pd.DataFrame(columns=['key','value'])



