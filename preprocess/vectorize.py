from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from abc import abstractmethod
from common import Nameable


class Vectorizer(Nameable):
    def __init__(self, name):
        super(Vectorizer, self).__init__(name=name)

    @abstractmethod
    def train(self, train_data):
        pass

    @abstractmethod
    def transform(self, train_data):
        pass


class Tfidf(Vectorizer):
    def __init__(self):
        super(Tfidf, self).__init__('tfidf')
        self.tfidf = TFIDF(
                min_df=5,
                max_features=None,
                ngram_range=(1, 3),
                use_idf=1,
                smooth_idf=1
                )

    def train(self, train_data):
        self.tfidf.fit(train_data)
        return self.tfidf

    def transform(self, data):
        return self.tfidf.transform(data)
