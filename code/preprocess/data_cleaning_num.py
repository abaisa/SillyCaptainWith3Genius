import pandas as pd
import numpy as np
from collections import OrderedDict
import scipy.stats as stats
import re
import random
import math


'''
清洗训练集和测试集中的数值特征
'''

#观察有哪些特征需要清洗
def cleaning_feature(x_train):
    count = 0
    col_list = []
    fearture_name = x_train.columns
    for col in fearture_name:
        try:
            x_train.ix[:, col].astype(float)
        except Exception as e:
            count = count + 1
            col_list.append(col)
    print("需要清洗的数值特征数：%d" %count)  # 训练集总共有168个数值特征需要进行清洗
    return col_list


#观察数值特征特殊的表达种类
def special_num(x_train,col_list):
    m=len(x_train)
    for col in col_list:
        kind_dict =dict()
        for i in range(m):
            s=x_train.ix[i,col]
            try:
                float(s)
                if s is np.nan:
                    if s not in kind_dict.keys():
                        kind_dict[s] = 1
                    else:
                        kind_dict[s] = kind_dict[s] + 1
            except Exception as e:
                if s not in kind_dict.keys():
                    kind_dict[s] = 1
                else:
                    kind_dict[s] = kind_dict[s] + 1
        print(col,OrderedDict(sorted(kind_dict.items(), key=lambda t: t[1],reverse=True)))

#缺失值的情况
def missing_condition(features):
    missing_count = features.isnull().sum()[features.isnull().sum() > 0].sort_values(ascending=True)  # sum()按列求和，返回一个index为之前列名的Series,sort_values进行排序，ascending=True表示是升序排列
    missing_percent = missing_count / len(features) #计算缺失值占总样本的比例
    drop_count=missing_count[missing_percent>=0.99] #去除缺失比例为1的特征
    drop_list=drop_count.index
    # missing_df = pd.concat([drop_count, missing_percent],join='inner', axis=1, keys=['count', 'percent'])
    # print(missing_df)
    return list(drop_list)

#提取数值特征中的数字
def get_num(s):
    try:
        float(s)
    except Exception as e:
        if s=='未见': #对于未见这个词用0代替
            s=0
        s=re.findall(r'\d+\.?\d*',s) #找到所有的数字
        if s!=[]:
            s=np.mean([float(i) for i in s]) #取平均值
        else:
            s=np.nan
    finally:
        return s

def yanya(s): #眼压正常范围是10-21
    try:
        n=float(s)
        if s is np.nan:
            return np.nan
    except Exception as e:
        n = re.search('\d+\.?\d*', s)
        if n == None:
            if '正常' in s:
                n=15
            elif '高' in s:
                n=25
            else:
                return np.nan
        else:
            n = n.group(0)
    n = float(n)
    if (n < 10):
        return 1
    elif (n > 21):
        return 3
    else:
        return 2

def feature_2409(s):
    try:
        float(s)
    except Exception as e:
        s=re.match('\d+\.?\d*',s)
        s=s.group(0)
    finally:
        return s

def most_value(series):
    num_list = []
    for s in series:
        try:
            s = float(s)
            if math.isnan(s) == False:  # 判断是否是nan
                num_list.append(s)
        except Exception as e:
            s=get_num(s)  #把数字获得
            if s is not np.nan:
                num_list.append(s)
            else:
                continue
    # print(num_list)
    return stats.mode(num_list)[0][0], np.percentile(num_list, 1), np.percentile(num_list, 99) # 众数，分位数

def feature_0425(s,mode,min,max):
    normal_list=['正常','异常']
    slow_list=['缓慢']
    fast_list=['急促']
    cucao_list=['粗糙']
    try:
        float(s)
        return s
    except Exception as e:
        for string in normal_list:
            if string in s:
                return mode  #正常用众数填充
        for string in slow_list:
            if string in s:
                return min #缓慢用较小的数填充
        for string in fast_list:
            if string in s:
                return max  #急促用较大的数填充
        for string in cucao_list:
            return 21 #粗糙用21填充，较大
        s = re.search('\d+\.?\d*', s)  # 对于没有检查的，匹配不到数字，也会返回np.nan
        if s == None:
            return np.nan
        else:
            s = s.group(0)
            return s

def feature_2413(s):
    try:
        float(s)
        return s
    except Exception as e:
        return np.nan

def feature_1002(s): #是否有心率疾病,属于中文特征
    if s is np.nan or 'nan' in s:
        return '缺失'
    elif '异常' in s or '正常' in s:
        return '正常'
    elif '不齐' in s or '波' in s or '偏' in s or '早搏' in s or '阻滞' in s:
        return '异常'
    elif '缓' in s:
        return '缓'
    elif '速' in s:
        return '速'
    else:
        return '其他'


def feature_0424(s): #心率,正常范围为60-100
    try:
        float(s)
    except Exception as e:
        if '正常' in s or '异常' in s: #将s设在正常范围内的数值
            s=72
        elif '缓' in s: #将s设在过缓范围内的数值
            s=50
        elif '速' in s:#将s设在过速范围内的数值
            s=110
        else:
            s = re.findall(r'\d+\.?\d*', s)  # 找到所有的数字
            if s != []:
                s = np.mean([float(i) for i in s])  # 取平均值
            else:
                s=np.nan
    finally:
        s=float(s)
        if s is np.nan:
            return 2  #没有则认为是正常的
        elif s<60: #过缓，标记为1
            return 1
        elif s>100: #过快，标记为3
            return 3
        else:     #正常，标记为2
            return 2

def shili(s): #视力的正常范围是等于或大于1.0
    try:
        float(s)
    except Exception as e:
        unnormal_list=['指数','光感','手动','义眼','失明','无光感']
        if  '指数' in s or '光感' in s or '手动' in s or '义眼' in s or '失明' in s or '无光感' in s :
            s=0.01
        elif '正常' in s:
            s=random.choice([1.0,1.2,1.5,2.0]) #在正常值中随机选择
        else:
            s = re.findall(r'\d+\.?\d*', s)  # 找到所有的数字
            if s != []:
                s = np.mean([float(i) for i in s])  # 取平均值
            else:
                s = np.nan
    finally:
        return s

def feature_1334(s): #中文特征
    if s is np.nan or 'nan' in s:
        return '缺失'
    elif '动脉' in s or '血压' in s or '糖尿病' in s:
        return '血压'
    elif '正常' in s or '异常' in s:
        return '正常'
    else:
        return '其他病变'

def contain(x_train):
    columns=x_train.columns
    length=len(x_train)
    for col in columns:
        for i in range(length):
            s=str(x_train.loc[i,col])
            if '阴性' in s or '阳性' in s or '+' in s:
                print(col)
                break

def yang(s): #根据阳性+的个数返回数字
    s=str(s)
    if '阳性' in s:
        return 1
    elif '+' in s:
        return s.count("+")
    else:
        return 0

#特征处理
def feature_cleaning_num(df,drop_columns, yang_columns, common_columns):
    #删除不重要的特征
    df.drop(drop_columns, axis=1, inplace=True)

    # 阳性列处理
    for col in yang_columns:
        df[col + '_阳性'] = df[col].map(yang)

    # 特殊数值特征处理
    df['1319'] = df['1319'].map(yanya)  # 右眼压特征的处理，离散化为高，低，正常,用1,2,3表示
    df['1320'] = df['1320'].map(yanya)  # 左眼眼压
    df[['1321', '1322', '1325', '1326']] = df[['1321', '1322', '1325', '1326']].applymap(shili)  # 视力
    df['2409'] = df['2409'].map(feature_2409)
    df.ix[df['0104'] == '心内各结构未见明显异常', '0104'] = df.ix[
        df['0104'] != '心内各结构未见明显异常', '0104'].median()  # 用中位数填充
    df['0424'] = df['0424'].map(feature_0424)
    mode_0425, min_0425, max_0425 = most_value(df['0425'])
    df['0425'] = df['0425'].map(lambda s: feature_0425(s, mode_0425, min_0425, max_0425))
    df['2413'] = df['2413'].map(feature_2413)

    # 特殊中文特征
    df['1002'] = df['1002'].map(feature_1002)
    df['1334'] = df['1334'].map(feature_1334)
    df_encoding = pd.get_dummies(df[['1002', '1334']])
    df.drop(['1002', '1334'], axis=1, inplace=True)

    # 普通特征
    df.ix[:, common_columns] =df.ix[:, common_columns].applymap(get_num)

    return df, df_encoding

# 训练集数值特征清洗
def train_data_cleaning_num(drop_columns, special_columns, yang_columns):
    train_data = pd.read_csv('../../data/train_num_uncleaning.csv', encoding='utf-8')
    x_train = train_data.ix[:, 6:]
    y_train = train_data.ix[:, :6]
    train_col_list = cleaning_feature(x_train)  # 得到需要清洗的特征名列表,总共214个

    train_common_columns = []
    for col in train_col_list:
        if col not in drop_columns and col not in special_columns:
            train_common_columns.append(col)

    # 特征处理
    x_train, x_train_encoding = feature_cleaning_num(x_train, drop_columns, yang_columns, train_common_columns)
    train_data=pd.concat([y_train, x_train, x_train_encoding], axis=1)

    print("清洗完后，训练集列数：%d" % len(train_data.columns))  # 421
    train_data.to_csv('../../data/train_data_num_final.csv', index=False, encoding='utf-8')

# 测试集数值特征清洗
def test_data_cleaning_num(drop_columns, special_columns, yang_columns):
    test_data = pd.read_csv('../../data/test_num_uncleaning.csv', encoding='utf-8')
    x_test = test_data.ix[:, 1:]
    id_test = test_data.ix[:, 0]
    test_col_list = cleaning_feature(x_test)  # 得到需要清洗的特征名列表,测试集有134个数值特征需要清洗

    test_common_columns = []
    for col in test_col_list:
        if col not in drop_columns and col not in special_columns:
            test_common_columns.append(col)

    # 特征处理
    x_test, x_test_encoding = feature_cleaning_num(x_test, drop_columns, yang_columns, test_common_columns)
    test_data=pd.concat([id_test, x_test, x_test_encoding], axis=1)

    print("测试集列数：%d" % len(test_data.columns))  # 416
    test_data.to_csv('../../data/test_data_num_final.csv', index=False, encoding='utf-8')

def data_cleaning_num_run():
    drop_columns = ['0214', '1104', '1335']  # 不重要的特征
    special_columns = ['1319', '1320', '1321', '1322', '1325', '1326', '1334', '0104', '0424', '0425', '1002', '1334',
                       '2409', '2413']  # 进行特殊处理的特征
    yang_columns = ['2376', '300151', '819007', 'I49012']  # 增加指明阳性的列
    # yinyang_total_columns = ['1171', '139', '1474', '2177', '2247', '2371', '2376', '300017', '300093', '300099',
    #                         '300151', '300152', '312', '3193', '809025', '819007', '819018', 'I49012']

    # 训练集处理
    train_data_cleaning_num(drop_columns, special_columns, yang_columns)
    # 测试集处理
    test_data_cleaning_num(drop_columns, special_columns, yang_columns)

if __name__ == '__main__':
    data_cleaning_num_run()




