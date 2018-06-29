from preprocess.cut import Jieba, Thulac, Cleaner
from preprocess.vectorize import Tfidf
from model.model_trainer import Svm, Xgboost
import data
from sklearn.externals import joblib


def train_pipeline(cut, vectorizer, model_trainer, record_num=None):
    print('reading...')
    alltext, accu_label, law_label, time_label = data.read_trainData("./data/data_train.json", record_num)

    print('cutting...')
    train_text = cut.cut(alltext)
    joblib.dump(train_text, './data/{}_cut_train.txt'.format(cut.name))

    print('cleaning...')
    cleaner = Cleaner()
    cleaned_train_text = cleaner.clean(train_text)
    joblib.dump(cleaned_train_text, './data/{}_cut_train_cleaned.txt'.format(cut.name))

    vectorizer_name = '{}_{}'.format(cut.name, vectorizer.name)
    print('{} training...'.format(vectorizer_name))
    vectorizer = vectorizer.train(cleaned_train_text)
    joblib.dump(vectorizer,
                './model/{}/predictor/model/{}_vectorizer.model'.format(model_trainer.name, vectorizer_name))

    print('{} vectorizing...'.format(vectorizer))
    vec = vectorizer.transform(cleaned_train_text)
    joblib.dump(vec, './data/vec_{}.txt'.format(vectorizer_name))

    print('acc training...')
    model = model_trainer.train(vec, accu_label)
    joblib.dump(model, './model/{}/predictor/model/{}_acc.model'.format(model_trainer.name, vectorizer_name))

    print('law training...')
    model = model_trainer.train(vec, law_label)
    joblib.dump(model, './model/{}/predictor/model/{}_law.model'.format(model_trainer.name, vectorizer_name))

    print('time training...')
    model = model_trainer.train(vec, time_label)
    joblib.dump(model, './model/{}/predictor/model/{}_time.model'.format(model_trainer.name, vectorizer_name))

