import numpy as np
from lightgbm.sklearn import LGBMRegressor

def error_fun(pret,target):
    a =np.log(pret+1)
    b = np.log(target+1)
    c = b - a
    d = pow(c,2)
    e = sum(d) / len(a)
    error = e
    return 'error',float(error)

'''
    X, Y 为原始的训练集和标注
'''
def get_useful_features_byLightBGM(X, Y):
    # 特殊参数设置
    importance_filter = 6

    model_3 = LGBMRegressor(num_leaves=36, n_estimators=100, learning_rate=0.07, random_state=0)
    Y_log = np.log1p(Y)
    model_3.fit(X, Y_log, verbose=True)

    feature_score = model_3.feature_importances_
    importance_feature_map = list(zip(feature_score, X.columns))

    useless_feature = []
    for i in importance_feature_map:
        if i[0] <= importance_filter:
            useless_feature.append(i[1])
    feature = [c for c in X.columns]
    useful_feature = [aa for aa in feature if aa not in useless_feature]
    print('有用：',len(useful_feature))
    print('无用：',len(useless_feature))
    print('全部：',len(feature))

    return useful_feature


import pickle #pickle模块

'''
保存 和 载入 model
'''
def save_model(model, info):
    with open('model/' + str(info) + '.pickle', 'wb') as f:
        pickle.dump(model, f)

def load_model(info):
    with open('model/' + str(info) + '.pickle', 'rb') as f:
        return pickle.load(f)