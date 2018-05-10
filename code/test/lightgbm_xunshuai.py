import gc

import numpy as np
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import train_test_split

from common.data_fun import train

train_ = train
# test_ = pd.read_csv('test.csv')

gg = [c for c in train_.columns if c not in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']] #102列
# test_data = test_[gg] #102
features = [feature for feature in gg if feature != 'vid']

train_X = train_[features]
# test_X = test_data[features]

def error_(pret,target):
    a =np.log(pret+1)
    b = np.log(target+1)
    c = b - a
    d = pow(c,2)
    e = sum(d) / len(a)
    error = e
    return 'error',float(error)

for i in range(10):
    #####################################################3#########
    train_y_1 = train_['收缩压']
    model_1 = LGBMRegressor(learning_rate=0.07, num_leaves=41, n_estimators=110, random_state=0)
    model_1.fit(train_X, train_y_1.values.ravel(), verbose=True)

    feature_score = model_1.feature_importances_
    feature_importances = zip(feature_score,train_X.columns)
    feature_importances = sorted(feature_importances)

    unuseful_feature = []
    for i in feature_importances:
        if i[0] == 0:
            unuseful_feature.append(i[1])
    use_features = [aa for aa in features if aa not in unuseful_feature]
    print('有用：',len(use_features))
    print('无用：',len(unuseful_feature))
    print('全部：',len(features))

    train_X_1 = train_[use_features]

    x1, x2, y1, y2 = train_test_split(train_X_1, train_y_1, test_size=0.2)

    model_1 = LGBMRegressor(learning_rate=0.07,num_leaves=41,n_estimators=110,random_state=0)
    model_1.fit(x1,y1.values.ravel(),verbose=True)
    val_1 = model_1.predict(x2)
    '''
    preds_1 = model_1.predict(test_X)
    '''
    print(error_(val_1,y2))
    val_1_error = error_(val_1,y2)

    del x1, x2, y1, y2
    gc.collect()

    ##############################################################
    train_y_2 = train_['舒张压']

    model_2 = LGBMRegressor(num_leaves=36,n_estimators=140,random_state=0,learning_rate=0.06)
    model_2.fit(train_X, train_y_2.values.ravel(), verbose=True)

    feature_score = model_2.feature_importances_
    feature_importances = zip(feature_score,train_X.columns)
    feature_importances = sorted(feature_importances)

    unuseful_feature = []
    for i in feature_importances:
        if i[0] == 0:
            unuseful_feature.append(i[1])
    use_features = [aa for aa in features if aa not in unuseful_feature]
    print('有用：',len(use_features))
    print('无用：',len(unuseful_feature))
    print('全部：',len(features))

    train_X_2 = train_[use_features]

    x1, x2, y1, y2 = train_test_split(train_X_2, train_y_2, test_size=0.2)
    model_2 = LGBMRegressor(num_leaves=36,n_estimators=140,random_state=0,learning_rate=0.06)
    model_2.fit(x1, y1.values.ravel(), verbose=True)

    val_2 = model_2.predict(x2)
    '''
    preds_2 = model_2.predict(test_X)
    '''
    print(error_(val_2, y2))
    val_2_error = error_(val_2, y2)

    del x1, x2, y1, y2
    gc.collect()

    ###################################################################
    train_y_3 = train_['血清甘油三酯']


    model_3 = LGBMRegressor(num_leaves=36, n_estimators=100, learning_rate=0.07, random_state=0)
    model_3.fit(train_X, train_y_3.values.ravel(), verbose=True)

    feature_score = model_3.feature_importances_
    feature_importances = zip(feature_score,train_X.columns)
    feature_importances = sorted(feature_importances)

    unuseful_feature = []
    for i in feature_importances:
        if i[0] <= 6:
            unuseful_feature.append(i[1])
    use_features = [aa for aa in features if aa not in unuseful_feature]
    print('有用：',len(use_features))
    print('无用：',len(unuseful_feature))
    print('全部：',len(features))

    train_X_3 = train_[use_features]

    x1, x2, y1, y2 = train_test_split(train_X_3, train_y_3, test_size=0.2)
    model_3 = LGBMRegressor(num_leaves=36,n_estimators=100,learning_rate=0.07,random_state= 0)
    model_3.fit(x1, y1.values.ravel(), verbose=True)

    val_3 = model_3.predict(x2)
    '''
    preds_3 = model_3.predict(test_X)
    '''
    print(error_(val_3, y2))
    val_3_error = error_(val_3, y2)

    del x1, x2, y1, y2
    gc.collect()
    ####################################################
    train_y_4 = train_['血清高密度脂蛋白']

    model_4_ = LGBMRegressor(n_estimators=100, num_leaves=101, random_state=0, learning_rate=0.1)
    model_4_.fit(train_X, train_y_4.values.ravel(), verbose=True)

    feature_score = model_3.feature_importances_
    feature_importances = zip(feature_score,train_X.columns)
    feature_importances = sorted(feature_importances)

    unuseful_feature = []
    for i in feature_importances:
        if i[0] == 0:
            unuseful_feature.append(i[1])
    use_features = [aa for aa in features if aa not in unuseful_feature]
    print('有用：',len(use_features))
    print('无用：',len(unuseful_feature))
    print('全部：',len(features))
    train_X_4 = train_[use_features]

    x1, x2, y1, y2 = train_test_split(train_X_4, train_y_4, test_size=0.2)
    model_4 = LGBMRegressor(n_estimators=100,num_leaves=101,random_state=0,learning_rate=0.1)
    model_4.fit(x1, y1.values.ravel(), verbose=True)

    val_4 = model_4.predict(x2)
    '''
    preds_4 = model_4.predict(test_X)
    '''
    print(error_(val_4, y2))
    val_4_error = error_(val_4, y2)

    del x1, x2, y1, y2
    gc.collect()


    ##########################################################
    train_y_5 = train_['血清低密度脂蛋白']

    model_5 = LGBMRegressor(learning_rate=0.06,num_leaves=121,n_estimators=100,random_state=0)
    model_5.fit(train_X, train_y_5.values.ravel(), verbose=True)


    feature_score = model_5.feature_importances_
    feature_importances = zip(feature_score,train_X.columns)
    feature_importances = sorted(feature_importances)
    # print(feature_score)
    # print(list(feature_importances))

    unuseful_feature = []
    for i in feature_importances:
        if i[0] == 0:
            unuseful_feature.append(i[1])
    #
    # print(unuseful_feature)
    # print(len(unuseful_feature))

    use_features = [aa for aa in features if aa not in unuseful_feature]
    print('有用：',len(use_features))
    print('无用：',len(unuseful_feature))
    print('全部：',len(features))
    '''
    train_X_5 = train_[use_features]
    '''
    #train_y_5 = train_['血清低密度脂蛋白']

    x1, x2, y1, y2 = train_test_split(train_X, train_y_5, test_size=0.2)
    model_5 = LGBMRegressor(learning_rate=0.06,num_leaves=121,n_estimators=100,random_state=0)
    model_5.fit(x1, y1.values.ravel(), verbose=True)

    val_5 = model_5.predict(x2)

    print(error_(val_5, y2))
    val_5_error = error_(val_5, y2)
    del x1, x2, y1, y2
    gc.collect()


    print('total error: ',(val_1_error[1]+val_2_error[1] + val_3_error[1] + val_4_error[1] + val_5_error[1]) / 5)