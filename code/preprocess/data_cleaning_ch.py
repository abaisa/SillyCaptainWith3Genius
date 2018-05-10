import pandas as pd
import numpy as np
from collections import OrderedDict
import re
import math

'''
清洗训练集和测试集中的中文特征，目前只处理部分
'''

#观察中文特征有哪些不同的种类
def kind_num(x_train,col_list):
    m=len(x_train)
    for col in col_list:
        kind_dict =dict()
        for i in range(m):
            s=x_train.ix[i,col]
            if s not in kind_dict.keys():
                kind_dict[s] = 1
            else:
                kind_dict[s] = kind_dict[s] + 1
        print(col,OrderedDict(sorted(kind_dict.items(), key=lambda t: t[1],reverse=True)))

#观察数值特征特殊的表达种类
def special_num(x_train,col_list):
    m=len(x_train)
    for col in col_list:
        kind_dict =dict()
        for i in range(m):
            s=x_train.ix[i,col]
            try:
                float(s)
            except Exception as e:
                if s not in kind_dict.keys():
                    kind_dict[s] = 1
                else:
                    kind_dict[s] = kind_dict[s] + 1
        print(col,OrderedDict(sorted(kind_dict.items(), key=lambda t: t[1],reverse=True)))

def feature_0409(series): #根据有无病史进行划分,五种情况
    feature_0409_encoding=pd.DataFrame(data=np.zeros( (len(series),5) ),index=series.index,columns=['0409_血压','0409_血糖','0409_血脂','0409_无病史', '0409_其他病史']) #空的dataframe
    for i in series.index:
        s=series[i]
        if s is np.nan:
            feature_0409_encoding.loc[i,'0409_无病史']=1
        else:
            if '血压' in s:
                feature_0409_encoding.loc[i, '0409_血压'] = 1
            if '糖尿病' in s or '血糖' in s:
                feature_0409_encoding.loc[i, '0409_血糖'] = 1
            if '脂肪肝' in s or '血脂' in s:
                feature_0409_encoding.loc[i, '0409_血脂'] = 1
            if ('异常' in s or '正常' in s) and (feature_0409_encoding.loc[i, '0409_血压'] == 0 and feature_0409_encoding.loc[i, '0409_血糖'] == 0 and feature_0409_encoding.loc[i, '0409_血脂'] == 0):
                feature_0409_encoding.loc[i, '0409_无病史'] = 1
            if feature_0409_encoding.loc[i, '0409_血压'] == 0 and feature_0409_encoding.loc[i, '0409_血糖'] == 0 and feature_0409_encoding.loc[i, '0409_血脂'] == 0 and feature_0409_encoding.loc[i, '0409_无病史'] == 0:
                feature_0409_encoding.loc[i, '0409_其他病史'] = 1
    return feature_0409_encoding

def feature_0434(series):
    feature_0434_encoding=pd.DataFrame(data=np.zeros( (len(series),5) ),index=series.index,columns=['0434_血压','0434_血糖','0434_血脂','0434_无病史', '0434_其他病史']) #空的dataframe
    for i in series.index:
        s=series[i]
        if s is np.nan:
            feature_0434_encoding.loc[i,'0434_无病史']=1
        else:
            if '血压' in s:
                feature_0434_encoding.loc[i, '0434_血压'] = 1
            if '糖尿病' in s or '血糖' in s:
                feature_0434_encoding.loc[i, '0434_血糖'] = 1
            if '脂肪肝' in s or '血脂' in s:
                feature_0434_encoding.loc[i, '0434_血脂'] = 1
            if ('无' in s or '健康' in s) and (feature_0434_encoding.loc[i, '0434_血压'] == 0 and feature_0434_encoding.loc[i, '0434_血糖'] == 0 and feature_0434_encoding.loc[i, '0434_血脂'] == 0):
                feature_0434_encoding.loc[i, '0434_无病史'] = 1
            if feature_0434_encoding.loc[i, '0434_血压'] == 0 and feature_0434_encoding.loc[i, '0434_血糖'] == 0 and feature_0434_encoding.loc[i, '0434_血脂'] == 0 and feature_0434_encoding.loc[i, '0434_无病史'] == 0:
                feature_0434_encoding.loc[i, '0434_其他病史'] = 1
    return feature_0434_encoding

def feature_1316(s):
    if s is np.nan:
        return '缺失'
    elif '动脉' in s or '血压' in s:
        return '血压'
    elif '糖尿病' in s:
        return '血糖'
    elif '正常' in s or '异常' in s:
        return '正常'
    else:
        return '其他病变'

def feature_1402(s):
    if s is np.nan:
        return '缺失'
    elif '快' in s:
        return '快'
    elif '慢' in s:
        return '慢'
    elif '异常' in s or ('正常' in s and '低于正常' not in s):
        return '正常'
    else:
        return '缺失'

def feature_4001(s): #血管特征分为缺失、正常和异常三种情况
    if s is np.nan:
        return '缺失'
    elif '减弱'in s or '硬化' in s:
        return '异常'
    elif '正常' in s or '异常' in s or '良好' in s:
        return '正常'
    else:
        return '缺失'

def feature_A705(s): #脂肪肝
    if s is np.nan:
        return '缺失'
    elif '异常' in s:
        return '正常'
    else:
        return '异常'

def feature_0912(s): #甲状腺分等级
    if s is np.nan or '不确定' in s or '可能' in s:
        return '不确定'
    elif '异常' in s or '不肿大' in s or '无肿大' in s:
        return '正常'
    elif '结节' in s:
        return '结节'
    elif '肿大' in s or '甲亢' in s:
        return '肿大'
    else:
        return '其他'

def get_num(s):  # 数值特征，获取数字,缺失值不用处理
    try:
        float(s)
    except Exception as e:
        s = re.findall(r'\d+\.?\d*', s)  # 找到所有的数字
        if s == []:  # 没有找到
            s = np.nan
        else:
            s = np.mean([float(i) for i in s])  # 取平均值
    finally:
        return s

def median(series): #中位数
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
    return  np.percentile(num_list, 50)

def feature_1363(s,median):
    try:
        float(s)
    except Exception as e:
        if s=='阴性' or s=='-':
            s=median
        else:
            s=get_num(s)
    finally:
        return s

def feature_30007(s):
    if s is np.nan:
        return np.nan
    elif 'Ⅰ' in s:
        return 1
    elif 'Ⅱ' in s or 'Ⅱ度' in s or 'II' in s or '中度' in s or '正常' in s or '未见异常' in s:
        return 2
    elif 'Ⅲ' in s or 'Ⅲ度' in s:
        return 3
    elif 'Ⅳ' in s or 'Ⅳ度' in s:
        return 4
    else:
        return np.nan

def feature_0102(s): #缺失或者没有脂肪肝指标的设为0，其余按照程度设为1,2,3
    if s is np.nan:
        return 0
    elif '肝:脂肪肝（轻度）' in s:
        return 1
    elif '肝:脂肪肝（中度）' in s:
        return 2
    elif '肝:脂肪肝（重度）' in s:
        return 3
    else:
        return 0

def feature_0113(s): #肝
    if s is np.nan:
        return '缺失'
    elif '增强' in s:
        return '增强'
    elif '正常' in s:
        return '正常'
    else:
        return '缺失'

def feature_0215(s): #咽部情况，分为缺失，正常和异常3种情况
    if s is np.nan:
        return '缺失'
    elif '正常' in s or '异常' in s:
        return '正常'
    else:
        return '异常'

def feature_0217(s): #扁桃体情况，分为缺失，正常，肿大（充血）和其4种情况
    if s is np.nan:
        return '缺失'
    elif '大' in s:
        return '肿大'
    elif '正常' in s or '异常' in s:
        return '正常'
    else:
        return '其他'

def sex(df_male,df_female):
    sex=pd.Series(index=df_male.index,data=np.zeros( (len(df_male),) ))
    for i in df_male.index:
        for col in df_male.iloc[:,:2].columns:
            s=df_male.loc[i,col]
            if s is not np.nan:
                sex[i]='男'
                break
        if sex[i]!='男':
            for col in df_male.iloc[:,2:].columns:
                s = str(df_male.loc[i, col])
                if '前列腺' in s:
                    sex[i] = '男'
                    break
        if sex[i]!='男':
            for col in df_female.iloc[:,:7].columns:
                s=df_female.loc[i,col]
                if s is not np.nan:
                    sex[i]='女'
                    break
        if sex[i] != '男' and sex[i] != '女':
            for col in df_female.iloc[:, 7:].columns:
                s = df_female.loc[i, col]
                s = str(s)
                if '乳腺' in s or '乳房' in s or '子宫' in s or '卵巢' in s:
                    sex[i] = '女'
                    break
        if sex[i]!='男' and sex[i]!='女':
            sex[i]='不知道'
    return sex

def feature_0114(s):
    if s is np.nan:
        return '缺失'
    elif '毛糙' in s:
        return '毛躁'
    elif '未显示胆囊回声' in s:
        return '无胆囊'
    else:
        return '正常'

def feature_0115(s):#胰腺
    if s is np.nan:
        return '缺失'
    elif '不清' in s or '欠清' in s:
        return '异常'
    else:
        return '正常'

def feature_0119(s):
    if s is np.nan:
        return '缺失'
    elif '欠' in s:
        return '异常'
    else:
        return '正常'

def feature_0203(s): #耳朵
    if s is np.nan:
        return '缺失'
    elif '正常' in s or '异常' in s:
        return '正常'
    elif '炎' in s or '浑浊' in s or '充血' in s:
        return '发炎'
    else:
        return '其他'

def feature_0209(s):
    if s is np.nan:
        return '缺失'
    elif '正常' in  s or '异常' in s:
        return '正常'
    elif '鼻炎' in s or '充血' in s:
        return '鼻炎'
    else:
        return '其他'

def feature_0210(s):
    if s is np.nan:
        return '缺失'
    elif '正常' in s or '异常' in s:
        return '正常'
    elif '鼻炎' in s:
        return '鼻炎'
    elif '大' in s:
        return '肿大'
    else:
        return '其他'

def feature_0501(s): #妇科检查
    if s is np.nan:
        return '缺失'
    elif '正常' in s or '异常' in s:
        return '正常'
    else:
        return '异常'

def feature_0503(s): #分泌物
    if s is np.nan:
        return '缺失'
    elif '多' in s:
        return '多'
    elif '中' in s:
        return '中'
    elif '充血' in s:
        return '充血'
    else:
        return '正常'

def feature_0509(s):
    if s is np.nan:
        return '缺失'
    elif '正常' in s or '异常' in s or '光滑' in s:
        return '正常'
    else:
        return '异常'

def feature_0516(s):
    if s is np.nan or '弃查' in s:
        return '缺失'
    elif '正常' in s or '异常' in s:
        return '正常'
    elif '萎缩小' in s:
        return '缩小'
    elif '增大' in s or '稍大' in s:
        return '增大'
    elif '缺' in s:
        return '切除'
    else:
        return '其他'

def feature_0911(s): #淋巴结
    if s is np.nan:
        return '缺失'
    elif '异常' in s or '不肿大' in s or '未触及' in s or '无肿大' in s:
        return '正常'
    elif '肿大'in  s or '结大' in s:
        return '肿大'
    else:
        return '正常'

def feature_0929(series): #乳腺
    feature_0929_encoding=pd.DataFrame(data=np.zeros( (len(series),6) ),index=series.index,columns=['0929_正常','0929_缺失','0929_小叶增生','0929_乳腺增生', '0929_结节','0929_其他']) #空的dataframe
    for i in series.index:
        s=series[i]
        if s is np.nan or '未查' in s or '弃查' in s:
            feature_0929_encoding.loc[i,'0929_缺失']=1
        elif '异常' in s:
            feature_0929_encoding.loc[i, '0929_正常'] = 1
        else:
            if '小叶增生' in s:
                feature_0929_encoding.loc[i, '0929_小叶增生'] = 1
            if '乳腺增生' in s:
                feature_0929_encoding.loc[i, '0929_乳腺增生'] = 1
            if '结节' in s:
                feature_0929_encoding.loc[i, '0929_结节'] = 1
            if feature_0929_encoding.loc[i,'0929_缺失']==0 and feature_0929_encoding.loc[i, '0929_正常']==0 and feature_0929_encoding.loc[i, '0929_小叶增生']==0 and feature_0929_encoding.loc[i, '0929_乳腺增生']==0 and feature_0929_encoding.loc[i, '0929_结节']==0:
                feature_0929_encoding.loc[i, '0929_其他'] = 1
    return feature_0929_encoding

def feature_0972(s):
    if s is np.nan or '未查' in  s or '弃查' in s or '不查' in s:
        return '缺失'
    elif '异常' in s:
        return '正常'
    elif '痔' in s:
        return '痔疮'
    else:
        return '其他'

def feature_0984(s): #前列腺
    if s is np.nan or '未查' in  s or '弃查' in s or '不查' in s:
        return '缺失'
    elif '异常' in s:
        return '正常'
    elif '增大' in s:
        return '增大'
    elif '中度增生' in s or '增生Ⅱ度' in s or '重度增生' in s:
        return '中度'
    elif '增生' in s:
        return '轻度'
    else:
        return '其他'

def feature_1001(s):
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

def feature_2501(s):
    if s is np.nan:
        return '缺失'
    elif '炎症' in s or 'Ⅱ' in s:
        return '炎症'
    elif '未见' in s or 'Ⅰ' in s:
        return '正常'
    elif '报告' in s:
        return '报告'
    else:
        return '其他'

def feature_3101(s):
    if s is np.nan:
        return '缺失'
    elif '脂肪肝' in s:
        return '脂肪肝'
    elif '异常' in s:
        return '正常'
    elif '报告' in s:
        return '报告'
    else:
        return '其他'

def feature_3301(s): #阴性和阳性
    if s is np.nan:
        return '缺失'
    elif '阴性' in s:
        return '阴性'
    elif '阳性' in s:
        return '阳性'
    else:
        return '其他'

def feature_3601(s): #骨量
    if s is np.nan:
        return '缺失'
    elif '正常' in s:
        return '正常'
    elif '减少' in s or '降低' in s or '疏松' in s:
        return '减少'
    else:
        return '正常'

def feature_3813(s): #肺活量
    if s is np.nan:
        return '缺失'
    elif '正常' in s or '良好' in s or '异常' in s or '肺活量及格' in s:
        return '正常'
    elif '限制' in s or '障碍' in s or '不及格' in s:
        return '异常'
    else:
        return '正常'

def yinyang(series,col): #分为阳性、阴性和缺失三种，无法确定情况的都认为是阴性
    yinyang_encoding=pd.DataFrame(data=np.zeros( (len(series),3) ),index=series.index,columns=[col+'_阴性',col+'_阳性',col+'_缺失']) #空的dataframe
    for i in series.index:
        s=str(series[i])
        if s is np.nan or 'nan' in s or '未做' in s:
            yinyang_encoding.loc[i, col + '_缺失'] = 1
        else:
            if '阴性' in s or '-' in s:
                yinyang_encoding.loc[i, col + '_阴性'] = 1
            if '阳性' in s:
                yinyang_encoding.loc[i, col + '_阳性'] = 1
            if '+' in s:
                yinyang_encoding.loc[i, col + '_阳性'] = s.count('+')
            if yinyang_encoding.loc[i, col + '_缺失'] ==0 and yinyang_encoding.loc[i, col + '_阴性'] ==0 and yinyang_encoding.loc[i, col + '_阳性'] == 0:
                yinyang_encoding.loc[i, col + '_阴性'] = 1
    return yinyang_encoding

def feature_300004(s): #判断是否是阳性，根据+的个数赋值
    s=str(s)
    if '阳性' in s:
        return 1
    elif "+" in s and re.search('\d',s)==None:
        return s.count('+')
    else:
        return 0

def feature_3400(s):
    if s is np.nan:
        return '缺失'
    elif '透明' in s:
        return '透明'
    elif '微混' in s or '微浑' in s:
        return '微混'
    elif '浑浊' in s or '混浊' in s:
        return '浑浊'
    else:
        return '缺失'

def feature_3426(s):
    if s is np.nan:
        return '缺失'
    elif '黄色' in s:
        return '黄色'
    elif '白浊' in s:
        return '白浊'
    elif '褐色' in s:
        return '褐色'
    else:
        return '缺失'

def create_yinyang(series,col): #特征300019和300036构造阴性列和阳性列，指明是阴性还是阳性
    yinyang_encoding = pd.DataFrame(data=np.zeros((len(series), 2)), index=series.index,columns=[col + '_阴性', col + '_阳性'])
    for i in series.index:
        s=str(series[i])
        if '阴性' in s or '-' in s:
            yinyang_encoding.loc[i, col + '_阴性'] = 1
        if '阳性' in s:
            yinyang_encoding.loc[i, col + '_阳性'] = 1
        if '+' in s:
            yinyang_encoding.loc[i, col + '_阳性'] = s.count("+")
    return yinyang_encoding

def feature_cleaning_ch(df, ch_columns, yinyang_columns, num_columns):
    # 构造性别特征
    df['sex'] = sex(df[['0120', '0984', '0954', '0102']], df[['0121', '0539', '0929', '0503', '0501', '0509', '0516', '0101', '0102', '0954']])  # 1360未知

    # 中文特征
    df['0114'] = df['0114'].map(feature_0114)
    df['0115'] = df['0115'].map(feature_0115)
    df['0119'] = df['0119'].map(feature_0119)
    df['0203'] = df['0203'].map(feature_0203)
    df['0209'] = df['0209'].map(feature_0209)
    df['0210'] = df['0210'].map(feature_0210)
    df['0501'] = df['0501'].map(feature_0501)
    df['0503'] = df['0503'].map(feature_0503)
    df['0509'] = df['0509'].map(feature_0509)
    df['0516'] = df['0516'].map(feature_0516)
    df['0911'] = df['0911'].map(feature_0911)
    df['0972'] = df['0972'].map(feature_0972)
    df['0984'] = df['0984'].map(feature_0984)
    df['1001'] = df['1001'].map(feature_1001)
    df['2501'] = df['2501'].map(feature_2501)
    df['3101'] = df['3101'].map(feature_3101)
    df['3301'] = df['3301'].map(feature_3301)
    df['3601'] = df['3601'].map(feature_3601)
    df['3813'] = df['3813'].map(feature_3813)
    df["0102"] = df["0102"].map(feature_0102)
    df["0113"] = df["0113"].map(feature_0113)
    df["0912"] = df["0912"].map(feature_0912)
    df["1316"] = df["1316"].map(feature_1316)
    df['1402'] = df['1402'].map(feature_1402)
    df['4001'] = df['4001'].map(feature_4001)
    df['A705'] = df['A705'].map(feature_A705)
    df['0215'] = df['0215'].map(feature_0215)
    df['0217'] = df['0217'].map(feature_0217)
    df['3400'] = df['3400'].map(feature_3400)
    df['3426'] = df['3426'].map(feature_3426)
    feature_0929_encoding = feature_0929(df['0929'])  # 乳腺的病
    feature_0409_encoding = feature_0409(df['0409'])  # 有无病史
    feature_0434_encoding = feature_0434(df['0434'])  # 有无病史
    df_encoding = pd.get_dummies(df[ch_columns])

    # 阴阳属性
    for col in yinyang_columns:
        yinyang_encoding = yinyang(df[col], col)
        df_encoding = pd.concat([df_encoding, yinyang_encoding], axis=1)

    # 300019和300036特征的阴阳列
    df_encoding = pd.concat([df_encoding, create_yinyang(df['300019'], '300019')], axis=1)
    df_encoding = pd.concat([df_encoding, create_yinyang(df['300036'], '300036')], axis=1)

    # 数值特征
    median_1363 = median(df['1363'])
    df['1363'] = df['1363'].map(lambda s: feature_1363(s, median_1363))
    df[['1873', '300019', '300036']] = df[['1873', '300019', '300036']].applymap(get_num)
    df['30007'] = df['30007'].map(feature_30007)
    df['300004'] = df['300004'].map(feature_300004)
    df = df[num_columns]

    return df, df_encoding, feature_0929_encoding, feature_0409_encoding, feature_0434_encoding

# 处理训练集的中文特征
def train_data_cleaning_ch(yinyang_columns, ch_columns, num_columns):
    train_data = pd.read_csv('../../data/train_ch_uncleaning.csv', encoding='utf-8')
    x_train = train_data.ix[:, 6:]
    y_train = train_data.ix[:, :6]

    # 特征处理
    x_train, x_train_encoding, feature_0929_encoding, feature_0409_encoding, feature_0434_encoding = feature_cleaning_ch(x_train, ch_columns, yinyang_columns, num_columns)
    train_data = pd.concat([y_train, x_train, x_train_encoding, feature_0929_encoding, feature_0409_encoding, feature_0434_encoding],axis=1)
    print("训练集列数：%d" % len(train_data.columns))  # 213
    train_data.to_csv('../../data/train_data_ch_final.csv', index=False, encoding='utf-8')

# 测试集的处理
def test_data_cleaning_ch(yinyang_columns, ch_columns, num_columns):
    test_data = pd.read_csv('../../data/test_ch_uncleaning.csv', encoding='utf-8')
    x_test=test_data.ix[:,1:]
    id_test=test_data.ix[:,0]

    # 特征处理
    x_test, x_test_encoding, feature_0929_encoding, feature_0409_encoding, feature_0434_encoding = feature_cleaning_ch(x_test, ch_columns, yinyang_columns, num_columns)
    test_data = pd.concat([id_test, x_test, x_test_encoding, feature_0929_encoding, feature_0409_encoding, feature_0434_encoding], axis=1)
    print("测试集列数：%d" % len(test_data.columns))  # 208
    test_data.to_csv('../../data/test_data_ch_final.csv', index=False, encoding='utf-8')

def data_cleaning_ch_run():
    # 阴阳特征
    yinyang_columns = ['100010', '2228', '2229', '2230', '2231', '2233', '2282', '229021', '30002', '3190', '3191','3192', '3194', '3195',
                       '3196', '3197', '3430', '3485', '3486', '3189']
    # 中文特征
    ch_columns = ['0114', '0115', '0119', '0203', '0209', '0210', '0501', '0503', '0509', '0516', '0911', '0972','0984', '1001', '2501', '3101', '3301',
                  '3601', '3813', '0102', '0113', '0912', '1316', '1402', '4001', 'A705', '0215', '0217', '3400', '3426', 'sex']
    # 数值特征
    num_cloumns=['1363', '1873', '300019', '300036', '30007', '300004']

    # 训练集处理
    train_data_cleaning_ch(yinyang_columns, ch_columns, num_cloumns)
    #测试集处理
    test_data_cleaning_ch(yinyang_columns, ch_columns, num_cloumns)


if __name__ == '__main__':
    data_cleaning_ch_run()
