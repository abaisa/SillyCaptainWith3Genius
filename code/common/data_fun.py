import numpy as np
import pandas as pd

train = pd.read_csv('../../data/train_data_num_ch_last.csv', encoding='utf-8')
test = pd.read_csv('../../data/test_data_num_ch.csv', encoding='utf-8')

train.index = train['vid']
train = train.drop('vid', axis=1)
train = train.astype(np.float32)
train_X = train.loc[:, "0104":]

y_names = ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']
train_Y_list = []
for i in range(5):
    train_Y_list.append(train.loc[:, y_names[i]])


# 训练集数据接口
def get_train_X():
    return train_X

def get_test_X():
    pass

# 修改返回值以对不同的Y进行训练
def get_train_Y():
    return train_Y_list[2]


def get_train_Y_by_name(name):
    y_name_dic = {
        '收缩压': 0,
        '舒张压': 1,
        '血清甘油三酯': 2,
        '血清高密度脂蛋白': 3,
        '血清低密度脂蛋白': 4
    }
    id = y_name_dic[name]
    return train_Y_list[id]