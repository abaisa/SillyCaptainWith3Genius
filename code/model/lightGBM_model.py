"""
    预测目标  血清甘油三酯
    lightBGM 单模型预测
"""

import numpy as np
from lightgbm.sklearn import LGBMRegressor
from common.my_fun import error_fun, get_useful_features_byLightBGM
from sklearn.model_selection import train_test_split

from common.data_fun import get_train_X, get_train_Y


def lightBGM_model_with_test(X, Y):
    model = LGBMRegressor(num_leaves=36, n_estimators=100, learning_rate=0.07, random_state=0)

    useful_feature = get_useful_features_byLightBGM(X, Y)
    X_U = X[useful_feature]

    x1, x2, y1, y2 = train_test_split(X_U, Y, test_size=0.2)
    y1_log = np.log1p(y1)
    model.fit(x1, y1_log, verbose=True)

    predict_log = model.predict(x2)
    predict = np.expm1(predict_log)
    error = error_fun(predict, y2)[1]

    del x1, x2, y1, y2
    return model, error

def lightBGM_model(X, Y):
    model = LGBMRegressor(num_leaves=36, n_estimators=100, learning_rate=0.07, random_state=0)
    model.fit(X, Y, verbose=True)
    return model

if __name__ == '__main__':
    best, worst, avg, rounds = 1, 0, 0, 10

    for i in range(rounds):
        print('round : ' + str(i))
        model, error = lightBGM_model_with_test(get_train_X(), get_train_Y())
        print('error : ' + str(error))

        if error < best:
            best = error
        if error > worst:
            worst = error
        avg += error

    avg /= rounds

    print('=========================')
    print('best > ' + str(best))
    print('worst > ' + str(worst))
    print('average > ' + str(avg))
