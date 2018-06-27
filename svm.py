from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
from predictor import data
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
import thulac
import time


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


def train_SVC(vec, label):
    SVC = LinearSVC()
    SVC.fit(vec, label)
    return SVC


if __name__ == '__main__':
    print('reading...')
    reading_start_time = time.time()
    alltext, accu_label, law_label, time_label = read_trainData('./data/data_train.json')
    print("reading finish in %s seconds", time.time() - reading_start_time)

    cut_start_time = time.time()
    print('cut text...')
    train_data = cut_text(alltext)
    print('cut text finish in %s seconds', time.time() - cut_start_time)
    joblib.dump(train_data, './cut_data_train.txt')

    tfidf_start_time = time.time()
    print('train tfidf...')
    tfidf = train_tfidf(train_data)
    print('train tfidf finish in %s seconds', time.time() - tfidf_start_time)

    vec = tfidf.transform(train_data)

    accu_svc_start_time = time.time()
    print('accu SVC')
    accu = train_SVC(vec, accu_label)
    print('accu SVC finish in %s seconds', time.time() - accu_svc_start_time)

    law_svc_start_time = time.time()
    print('law SVC')
    law = train_SVC(vec, law_label)
    print('law SVC finish in %s seconds', time.time() - law_svc_start_time)

    time_svc_start_time = time.time()
    print('time SVC')
    time_svc = train_SVC(vec, time_label)
    print('time SVC finish in %s seconds', time.time() - time_svc_start_time)

    print('saving model')
    save_start_time = time.time()
    joblib.dump(tfidf, 'predictor/model/tfidf.model')
    print('saving model tfidf in %s seconds', time.time() - save_start_time)

    save_start_time = time.time()
    joblib.dump(accu, 'predictor/model/accu.model')
    print('saving model accu in %s seconds', time.time() - save_start_time)

    save_start_time = time.time()
    joblib.dump(law, 'predictor/model/law.model')
    print('saving model law in %s seconds', time.time() - save_start_time)

    save_start_time = time.time()
    joblib.dump(time_svc, 'predictor/model/time.model')
    print('saving model time_svc in %s seconds', time.time() - save_start_time)



