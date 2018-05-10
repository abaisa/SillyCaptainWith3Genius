# -*- coding: utf-8 -*-
import operator

import pandas as pd
import xgboost as xgb

from common.data_fun import X, Y_all

if __name__ == '__main__':

    for Y in Y_all:
        xg_train = xgb.DMatrix(X, label=Y)

        params = {
        #     'min_child_weight': 100,
        #     'eta': 0.02,
        #     'colsample_bytree': 0.7,
        #     'max_depth': 7,
        #     'subsample': 0.7,
        #     'alpha': 1,
        #     'gamma': 1,
        #     'silent': 1,
        #     'verbose_eval': True,
        #     'seed': 12,
        #     'n_estimators': 100
        'max_depth': 7,
        'n_estimators': 100
        }
        rounds = 10

        bst = xgb.train(params, xg_train, num_boost_round=rounds)

        features = [x for x in X.columns if x != 'vid']

        importance = bst.get_score()
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        pd.DataFrame(importance).to_csv('./data/feature_importance/' + Y.name + '_weight.csv', index=False)


        importance = bst.get_score(importance_type = 'gain')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        pd.DataFrame(importance).to_csv('./data/feature_importance/' + Y.name + '_gain.csv', index=False)