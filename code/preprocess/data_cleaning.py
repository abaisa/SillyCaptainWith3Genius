import pandas as pd
import re

'''
提取出part1和part2的所有数值特征和中文特征，划分到训练集和测试集中
将训练集中各特征缺失比例高于或等于99%的特征进行去除，测试集中的这些特征也进行去除
再去除训练集中y值不规范的样本
'''

ch_pattern = re.compile(u'[\u4e00-\u9fa5]+')
num_pattern = re.compile(r'\d+')

# 判断字符串是否包含中文u'[\u4e00-\u9fa5]+'
def contain_ch(s,col):
    try:
        if ch_pattern.search(s):
            return True
        else:
            return False
    except Exception as e:
        print("无法进行正则匹配的列：" %col)

# 判断字符串是否包含数字
def contain_num(s,col):
    try:
        if num_pattern.search(s):
            return True
        else:
            return False
    except Exception as e:
        print("无法进行正则匹配的列：" % col)

# 读取原始数据
def file_read():
    part_1 = pd.read_csv('../../data/meinian_round1_data_part1_20180408.txt', delimiter='$', encoding='utf-8')
    part_2 = pd.read_csv('../../data/meinian_round1_data_part2_20180408.txt', delimiter='$', encoding='utf-8')
    y_train = pd.read_csv('../../data/meinian_round1_train_20180408.csv', encoding='gbk')
    y_test = pd.read_csv('../../data/meinian_round1_test_b_20180505.csv', encoding='gbk')
    y_train.set_index('vid', inplace=True)  # 将vid设为索引
    y_test.set_index('vid', inplace=True)

    return part_1, part_2, y_train, y_test

# 输出到文件
def file_write(train_data_num, test_data_num, train_data_ch, test_data_ch):
    train_data_num.to_csv('../../data/train_num_uncleaning.csv', encoding='utf-8')
    test_data_num.to_csv('../../data/test_num_uncleaning.csv', encoding='utf-8')
    train_data_ch.to_csv('../../data/train_ch_uncleaning.csv', encoding='utf-8')
    test_data_ch.to_csv('../../data/test_ch_uncleaning.csv', encoding='utf-8')

# 提取part1的数值和中文特征
def get_part1_num_ch(part_1, y_train, y_test):
    part1_feature = part_1.pivot_table(index='vid', columns='table_id', values='field_results',
                                       aggfunc=lambda x: ' '.join(str(v) for v in x))
    part1_ch_list = []  # part1中所有包含中文的特征
    for col in part1_feature.columns:
        series = part1_feature.loc[part1_feature[col].notnull(), col]
        s = series[-1]  # 以每一列的最后一个值的情况代表这一列是否含中文
        if contain_ch(s, col) == True:
            part1_ch_list.append(col)
    part1_num_feature = part1_feature.drop(part1_ch_list, axis=1)  # 去除part1中所有训练集的中文特征
    part1_train_num = pd.concat([y_train, part1_num_feature], axis=1, join='inner')  # part1中的训练集数据
    part1_test_num = pd.concat([y_test, part1_num_feature], axis=1, join='inner')  # part1中的测试集数据
    part1_test_num.drop(['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'], axis=1, inplace=True)  # 删除测试集中空的5项指标列

    # 提取part1的中文特征
    part1_ch_feature = part1_feature.loc[:, part1_ch_list]
    part1_train_ch = pd.concat([y_train, part1_ch_feature], axis=1, join='inner')
    part1_test_ch = pd.concat([y_test, part1_ch_feature], axis=1, join='inner')
    part1_test_ch.drop(['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'], axis=1, inplace=True)  # 删除5项指标列

    return part1_train_num, part1_test_num, part1_train_ch, part1_test_ch

def get_part2_num_ch(part_2, y_train, y_test):
    part2_feature = part_2.pivot_table(index='vid', columns='table_id', values='field_results',
                                       aggfunc=lambda x: ' '.join(str(v) for v in x))
    part2_num_list = []  # part2中所有包含数字的特征
    for col in part2_feature.columns:
        series = part2_feature.loc[part2_feature[col].notnull(), col]
        s = series[-1]
        if contain_num(s, col) == True:
            part2_num_list.append(col)
    part2_num_feature = part2_feature.loc[:, part2_num_list]
    part2_train_num = pd.concat([y_train, part2_num_feature], axis=1, join='inner')  # part2中的训练集数据
    part2_test_num = pd.concat([y_test, part2_num_feature], axis=1, join='inner')  # part2中的测试集数据
    part2_test_num.drop(['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'], axis=1, inplace=True)  # 删除测试集中空的5项指标

    # 提取part2的中文特征
    part2_ch_feature = part2_feature.drop(part2_num_list, axis=1)
    part2_train_ch = pd.concat([y_train, part2_ch_feature], axis=1, join='inner')
    part2_test_ch = pd.concat([y_test, part2_ch_feature], axis=1, join='inner')
    part2_test_ch.drop(['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'], axis=1, inplace=True)  # 删除5项指标列

    return part2_train_num, part2_test_num, part2_train_ch, part2_test_ch

# 划分part1和part2的特征为数值特征和中文特征
def split_feature(part_1, part_2, y_train, y_test):
    print("part1处理中......")
    part1_train_num, part1_test_num, part1_train_ch, part1_test_ch=get_part1_num_ch(part_1, y_train, y_test)
    print("part2处理中......")
    part2_train_num, part2_test_num, part2_train_ch, part2_test_ch=get_part2_num_ch(part_2, y_train, y_test)

    # 拼接part1和part2的数值特征
    part2_train_num.drop(['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'], axis=1, inplace=True)  # 删除训练中的5项指标，避免合并时重复
    train_num = pd.concat([part1_train_num, part2_train_num], axis=1)
    test_num = pd.concat([part1_test_num, part2_test_num], axis=1)

    # 拼接part1和part2的中文特征
    part2_train_ch.drop(['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'], axis=1, inplace=True)  # 删除训练中的5项指标，避免合并时重复
    train_ch = pd.concat([part1_train_ch, part2_train_ch], axis=1)
    test_ch = pd.concat([part1_test_ch, part2_test_ch], axis=1)

    return train_num, test_num, train_ch, test_ch

# 缺失值的情况
def missing_condition(features,rate):
    missing_count  =  features.isnull().sum()[features.isnull().sum() > 0].sort_values(ascending = True)  #  sum()按列求和，返回一个index为之前列名的Series,sort_values进行排序，ascending = True表示是升序排列
    missing_percent  =  missing_count / len(features) # 计算缺失值占总样本的比例
    drop_count = missing_count[missing_percent>= rate]
    drop_list = drop_count.index
    #  missing_df  =  pd.concat([drop_count, missing_percent],join = 'inner', axis = 1, keys = ['count', 'percent'])
    #  print(missing_df)
    return list(drop_list)

# 去除缺失值过高的特征
def delete_high_missing_feature(train_num, test_num, train_ch, test_ch):
    print("去除缺失值过高的特征....")
    x_train_num = train_num.iloc[:, 5:]  # 所有数值特征
    y_train_num = train_num.iloc[:, :5]  # 5项指标
    drop_list = missing_condition(x_train_num, 0.99)
    x_train_num.drop(drop_list, axis=1, inplace=True)  # 去除缺失比例为大于等于0.99的数值特征
    print("数值特征剩余个数：%d" % len(x_train_num.columns))  # 剩余406个数值特征
    train_data_num = pd.concat([y_train_num, x_train_num], axis=1)  # 训练集数字特征
    # 测试集也去除这些特征
    test_data_num = test_num.loc[:, x_train_num.columns]  # 测试集数字特征

    # 去除缺失值大于0.98的中文特征
    # 训练集
    x_train_ch = train_ch.iloc[:, 5:]  # 所有中文特征
    y_train_ch = train_ch.iloc[:, :5]  # 5项指标
    drop_list = missing_condition(x_train_ch, 0.98)
    x_train_ch.drop(drop_list, axis=1, inplace=True)  # 去除缺失比例大于等于0.98的特征
    print("中文特征剩余个数：%d" % len(x_train_ch.columns))  # 剩余187个中文特征
    train_data_ch = pd.concat([y_train_ch, x_train_ch], axis=1)  # 训练集中文特征
    # 测试集的处理
    test_data_ch = test_ch.loc[:, x_train_ch.columns]  # 保留与训练集相同的特征 ,测试集中文特征

    return train_data_num, test_data_num, train_data_ch, test_data_ch

# 去除不规范样本
def delete_unnormal_sample(y_train):
    drop_list = ['fa04c8db6d201b9f705a00c3086481b0',
              '7685d48685028a006c84070f68854ce1',
              '798d859a63044a8a5addf1f8c528629e',
              'd9919661f0a45fbcacc4aa2c1119c3d2',
              'de82a4130c4907cff4bfb96736674bbc',
              'bd0322cf42fc6c2932be451e0b54ed02']
    print('训练集样本个数：%d' % len(y_train))
    for i in y_train.index:
        try:
            y_train.loc[i, :].astype(float)
        except Exception as e:
            drop_list.append(i)
    print("去除y值不规范样本个数：%d" % len(drop_list))
    return drop_list

def data_cleaning_run():
    # 读取文件
    part_1, part_2, y_train, y_test = file_read()

    # 划分数值特征和中文特征
    train_num, test_num, train_ch, test_ch = split_feature(part_1, part_2, y_train, y_test)
    # 去除高缺失值后的数值特征和中文特征
    train_data_num, test_data_num, train_data_ch, test_data_ch = delete_high_missing_feature(train_num, test_num, train_ch, test_ch)
    # 去除训练集中y值不规范的样本
    drop_list=delete_unnormal_sample(train_data_num.iloc[:, :5])
    train_data_num.drop(drop_list, inplace=True)
    train_data_ch.drop(drop_list, inplace=True)

    # 输出文件
    file_write(train_data_num, test_data_num, train_data_ch, test_data_ch)

if __name__ ==  '__main__':
    data_cleaning_run()
















