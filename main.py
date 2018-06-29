from preprocess.cut import Jieba, Thulac, Cleaner
from preprocess.vectorize import Tfidf
from model.model_trainer import Svm, Xgboost
import data
from sklearn.externals import joblib
from evaluate import Evaluator
from common import ModelContext
from model.svm.predictor import Predictor as SvmPredictor
from model.xgboost.predictor import Predictor as XgboostPredictor


def train_pipeline(kind, cut, vectorizer, model_trainer, do_cut=False, do_vectorizer=False, record_num=None):
    print('reading...')
    alltext, accu_label, law_label, time_label = data.read_trainData("./data/data_train.json", record_num)

    if do_cut:
        print('cutting...')
        train_text = cut.cut(alltext)
        joblib.dump(train_text, './data/{}_cut_train.txt'.format(cut.name))

        print('cleaning...')
        cleaner = Cleaner()
        cleaned_train_text = cleaner.clean(train_text)
        joblib.dump(cleaned_train_text, './data/{}_cut_train_cleaned.txt'.format(cut.name))
    else:
        print('load existing cut file {}...'.format('./data/{}_cut_train_cleaned.txt'.format(cut.name)))
        cleaned_train_text = joblib.load('./data/{}_cut_train_cleaned.txt'.format(cut.name))

    vectorizer_name = '{}_{}'.format(cut.name, vectorizer.name)
    if do_vectorizer:
        print('{} training...'.format(vectorizer_name))
        vectorizer = vectorizer.train(cleaned_train_text)
        joblib.dump(vectorizer,
                    './model/{}/predictor/model/{}_vectorizer.model'.format(model_trainer.name, vectorizer_name))
        print('{} vectorizing...'.format(vectorizer))
        vec = vectorizer.transform(cleaned_train_text)
        joblib.dump(vec, './data/vec_{}.txt'.format(vectorizer_name))
    else:
        print('load existing vec file {}...'.format('./data/vec_{}.txt'.format(vectorizer_name)))
        vec = joblib.load('./data/vec_{}.txt'.format(vectorizer_name))

    print('{} training...'.format(kind))
    model = model_trainer.train(vec, accu_label)
    joblib.dump(model, './model/{}/predictor/model/{}_{}.model'.format(model_trainer.name, vectorizer_name, kind))


def train_model(kind, model_trainer, vec, label):
    print('{} training...'.format(kind))
    model = model_trainer.train(vec, label)
    joblib.dump(model, './model/{}/predictor/model/{}_{}.model'.format(model_trainer.name, vectorizer_name, kind))


def train_jieba_tfidf_svm(num_record=None):
    train_pipeline(Jieba(), Tfidf(), Svm(), num_record)


def train_thulac_tfidf_svm():
    train_pipeline(Thulac(), Tfidf(), Svm())


def train_jieba_tfidf_xgboost():
    # specify parameters via map
    kind = 'accu'
    param = {
        'max_depth': 5, 'learning_rate': 0.1, 'eta': 0.1, 'silent': True, 'objective': 'multi:softmax',
        'num_class': data.getClassNum(kind), 'n_estimators': 100, 'nthread': -1}
    num_round = 10
    train_pipeline(kind, Jieba(), Tfidf(), Xgboost(param, num_round))


def evaluate(predictor):
    test_file_name = 'data_valid.json'
    print('loading evaluator...')
    evaluator = Evaluator(predictor)
    print('output result...')
    evaluator.output_result(test_file_name)
    print('scoring...')
    score = evaluator.scoring(test_file_name)
    print(score)


if __name__ == '__main__':
    # train_jieba_tfidf_xgboost()
    evaluate(XgboostPredictor(Jieba(), './model/xgboost/predictor/model'))
    # evaluate(SvmPredictor(Jieba(), './model/svm/predictor/model'))
