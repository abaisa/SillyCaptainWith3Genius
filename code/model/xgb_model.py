import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from common.data_fun import X, Y_all

model_list = []
model_list.append({'n_estimators' : 150, 'max_depth' : 6, 'min_child_weight' : 4, 'colsample_bytree' : 0.7, 'subsample' : 0.7})
model_list.append({'n_estimators' : 180, 'max_depth' : 5, 'min_child_weight' : 3, 'colsample_bytree' : 1,   'subsample' : 0.9})
model_list.append({'n_estimators' : 240, 'max_depth' : 6, 'min_child_weight' : 1, 'colsample_bytree' : 0.6, 'subsample' : 0.9})
model_list.append({'n_estimators' : 240, 'max_depth' : 6, 'min_child_weight' : 1, 'colsample_bytree' : 0.6, 'subsample' : 0.9})
model_list.append({'n_estimators' : 200, 'max_depth' : 9, 'min_child_weight' : 2, 'colsample_bytree' : 0.9, 'subsample' : 0.8})

total = 0
cnt = 0

for Y in Y_all:
    # 划分数据集
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

    # 训练参数设置和执行
    params = model_list[cnt]

    cnt += 1
    rounds = 10

    # 训练
    xg_train = xgb.DMatrix(train_x, label = train_y)
    xgboost_model = xgb.train(params, xg_train)

    # 预测
    xg_test = xgb.DMatrix(test_x)
    xg_res = xgboost_model.predict(xg_test)

    # 计算指标
    acc = sum((np.log(xg_res + 1) - np.log(test_y + 1)) ** 2) / len(xg_res)
    print(Y.name + '--' + str(acc))
    total += acc

print('Final--' + str(total / 5))

