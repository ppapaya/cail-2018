from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
from datetime import datetime, date
from predictor import data
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
import thulac
import time
import xgboost as xgb
from sklearn.grid_search import GridSearchCV


dim = 5000
record_num = 10000

def cut_text(alltext):
    count = 0
    cut = thulac.thulac(seg_only = True)
    train_text = []
    for text in alltext:
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append(cut.cut(text, text = True))

    return train_text


def train_tfidf(train_data):
    tfidf = TFIDF(
            min_df = 5,
            max_features = dim,
            ngram_range = (1, 3),
            use_idf = 1,
            smooth_idf = 1
            )
    tfidf.fit(train_data)

    return tfidf


def read_trainData(path):
    fin = open(path, 'r', encoding = 'utf8')

    alltext = []

    accu_label = []
    law_label = []
    time_label = []

    line = fin.readline()
    count = 0
    while line and count <= record_num:
        # count += 1
        d = json.loads(line)
        alltext.append(d['fact'])
        accu_label.append(data.getlabel(d, 'accu'))
        law_label.append(data.getlabel(d, 'law'))
        time_label.append(data.getlabel(d, 'time'))
        line = fin.readline()
    fin.close()

    return alltext, accu_label, law_label, time_label


def train_xgboost(vec, label, kind):
    dtrain = xgb.DMatrix(vec, label)
    # specify parameters via map
    param = {'max_depth': 5, 'eta': 0.1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': data.getClassNum(kind)}
    num_round = 10
    return xgb.train(param, dtrain, num_round)


def cv_xgboost(vec, label, parameters):
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, nthread=-1, gamma=0,
                              objective='multi:softmax',min_child_weight=1, max_delta_step=0, subsample=1,
                              colsample_bytree=1,
                              colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                              seed=27,
                              missing=None)
    grid = GridSearchCV(model, parameters, cv=3, scoring='f1_macro', n_jobs=-1, refit='f1_macro',
                        verbose=2)
    grid.fit(vec, label)
    return grid


def cv_model(model_name, dtrain, dlabel, parameters):
    start_time = time.time()
    print(model_name + ' start')
    grid = cv_xgboost(dtrain, dlabel, parameters)
    print(model_name + ' finish', time.time() - start_time)
    joblib.dump(grid, 'predictor/model/' + model_name + '.model')
    print('------------------------------')
    print('Best Estimator for XGBoost')
    print(grid.best_estimator_)
    print('CV score is:', format(grid.best_score_, '.3f'))
    print('Best parameters set:')
    print(grid.best_estimator_.get_params())

def cv_params(label):
    # weight = float(len(label) - sum(label))/float(sum(label))
    return {
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 6, 10],
        # 'n_estimators': [200],
        'colsample_bytree': [0.5, 0.8, 1],
        'subsample': [0.5, 0.8, 1],
        'reg_alpha': [0, 1],
        'reg_lambda': [0, 1],
        # 'scale_pos_weight': [weight]
    }

if __name__ == '__main__':

    print('reading...')
    alltext, accu_label, law_label, time_label = read_trainData('./data/data_train.json')
    # train_data = joblib.load('./cut_data_train.txt')
    # tfidf = joblib.load('./predictor/model/tfidf.model')
    # vec = tfidf.transform(train_data)
    # joblib.dump(vec, './vec.text')

    vec = joblib.load('./vec.txt')
    # accu
    cv_model('xgboost_accu', vec, accu_label, cv_params(accu_label))

    # law
    cv_model('xgboost_law', vec, law_label, cv_params(law_label))

    # time
    cv_model('xgboost_time', vec, time_label, cv_params(time_label))
