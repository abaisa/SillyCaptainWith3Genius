import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor,DMatrix
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,BaggingRegressor,ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
# from lightgbm import LGBMRegressor
from sklearn.externals import joblib
y_names = ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白','血清低密度脂蛋白']
del_list=['fa04c8db6d201b9f705a00c3086481b0',
          '7685d48685028a006c84070f68854ce1',
          '798d859a63044a8a5addf1f8c528629e',
          'd9919661f0a45fbcacc4aa2c1119c3d2',
          'de82a4130c4907cff4bfb96736674bbc',]
#得到训练数据
train_data=pd.read_csv('./data/train_data_num_ch_last.csv',encoding='utf-8')
#删除异常的训练对象
for i in del_list:
    index_del=train_data[train_data['vid']==i].index
    train_data=train_data.drop(index_del,axis=0)
# train_data.to_csv('./data/csv_of_data/train_2.csv',index=False)


train_data.index=train_data['vid']
train_data=train_data.drop('vid',axis=1)
train_data=train_data.astype(np.float32)
train_x=train_data.loc[:,"0104":]

#
# def f(x):
#     if np.isnan(x):
#         return 1
#     else:
#         return 0
########################################
# train_x=train_x.applymap(f)
# train_x.to_csv('tmp.csv',index=False)
# a=pd.read_csv('tmp.csv',encoding='utf-8')
# # 缺失值比例大于0.9的列名
# drop_index=list(a.sum(0)[(a.sum(0)/38145)>0.96].index)
# print(len(drop_index))
# print(len(train_x.columns))
# # 删除缺失值
# train_x.drop(drop_index,axis=1,inplace=True)
############################################################
train_y=train_data.loc[:,y_names]


#划分数据集
train_x,test_x,train_y,test_y=train_test_split(train_x,train_y,test_size=0.3)
#提交的测试集
#测试集的特征
test_x_=pd.read_csv('./data/test_data_num_ch.csv',encoding='utf-8',na_values=np.nan)
#循环的测试id集合
test_names=test_x_['vid']
#index替换成id
test_x_.index=test_x_['vid']
test_x_=test_x_.drop('vid',axis=1).astype(np.float64)


#使用xgboost训练
###############################################
#n_estimators=100,max_depth=10
#
#
#
#
# clf=XGBRegressor(n_estimators=100,max_depth=7,min_child_weight=1,gamma=0)
# clf=LGBMRegressor(n_estimators=100,subsample_for_bin=50000,learning_rate=0.1)
# clf=BaggingRegressor()
result=[]
ERR=0
min_err=1

for j in [15000]:
    # m=pd.read_csv('./data/result/6_0.0322.csv',names=['vid','收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白','血清低密度脂蛋白'])




############################################

    # clf = LGBMRegressor(n_estimators=100, subsample_for_bin=j,learning_rate=0.08,num_leaves=46,subsample=0.97,min_split_gain=3)
    clf = XGBRegressor(n_estimators=200, max_depth=7, min_child_weight=1, gamma=0)
    for i in y_names:
        clf.fit(train_x, train_y[i])
        pred_ = clf.predict(test_x)
        a = sum((np.log(pred_ + 1) - np.log(test_y[i] + 1)) ** 2) / len(pred_)
        joblib.dump(clf, './data/model/{0}.model_{1}'.format(i, a))
        print(a)
        ERR += a
    print(ERR / 5.0)
#################################################
    # for i in y_names:
    #     if i=='血清甘油三酯':
    #         # clf=joblib.load('./data/model_select/血清甘油三酯.model_0.07747025561699392')
    #         pass
    #     elif i=='血清高密度脂蛋白':
    #         clf=joblib.load('./data/model/血清高密度脂蛋白.model_0.011323631892292683')
    #     elif i=='收缩压':
    #         clf=joblib.load('./data/model/收缩压.model_0.014185246052119297')
    #     elif i=='舒张压':
    #         clf = joblib.load('./data/model/舒张压.model_0.017727169499404994')
    #     else:
    #         clf=joblib.load('./data/model/血清低密度脂蛋白.model_0.03212948880505983')
    #     if i=='血清甘油三酯':
    #         pred=list(m['血清甘油三酯'])
    #         result.append(pred)
    #     else:
    #         pred=list(clf.predict(test_x_))
    #         result.append(pred)




    # 字典中的key值即为csv中列名
# dataframe = pd.DataFrame({'0':list(test_names),'1':result[0], '2':result[1],'3':result[2],'4':result[3],'5':result[4]})
# dataframe.index=dataframe['0']
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
# dataframe.to_csv("test.csv", sep=',',index=False,header=False)
#模型参数设置
# xlf = xgb.XGBRegressor(max_depth=10,
#                         learning_rate=0.1,
#                         n_estimators=10,
#                         silent=True,
#                         objective='reg:linear',
#                         nthread=-1,
#                         gamma=0,
#                         min_child_weight=1,
#                         max_delta_step=0,
#                         subsample=0.85,
#                         colsample_bytree=0.7,
#                         colsample_bylevel=1,
#                         reg_alpha=0,
#                         reg_lambda=1,
#                         scale_pos_weight=1,
#                         seed=1440,
#                         missing=None)