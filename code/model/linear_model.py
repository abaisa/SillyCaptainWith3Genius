'''
    预测目标  血清甘油三酯
    LinearRegression 单模型预测

    best > 0.075943681872589
    worst > 0.20423074141359016
    average > 0.12744208882408292
'''

import numpy as np
from common.my_fun import error_fun, get_useful_features_byLightBGM
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from common.data_fun import get_train_X, get_train_Y


def linearModel(X, Y):
    model = linear_model.LinearRegression()

    useful_feature = get_useful_features_byLightBGM(X, Y)
    X_U = X[useful_feature]

    x1, x2, y1, y2 = train_test_split(X_U, Y, test_size=0.2)
    y1_log = np.log1p(y1)
    model.fit(x1, y1_log)

    predict_log = model.predict(x2)
    predict = np.expm1(predict_log)
    error = error_fun(predict, y2)[1]

    return model, error

if __name__ == '__main__':
    train_x_full = get_train_X().fillna(train_x.median())
    train_y = get_train_Y()
    best, worst, avg, rounds = 1, 0, 0, 10

    for i in range(rounds):
        print('round : ' + str(i))
        error = linearModel(train_x_full, train_y)
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
