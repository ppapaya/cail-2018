from abc import abstractmethod
from string import punctuation
import thulac
import jieba
from common import Nameable


class Cut(Nameable):
    def __init__(self, name):
        super(Cut, self).__init__(name=name)

    def cut(self, all_text):
        count = 0
        train_text = []
        for text in all_text:
            count += 1
            if count % 2000 == 0:
                print(count)
            train_text.append(self.cut_text(text))
        return train_text

    @abstractmethod
    def cut_text(self, text):
        pass


class Jieba(Cut):
    def __init__(self):
        super(Jieba, self).__init__(name='jieba')

    def cut_text(self, test_sent):
        result = jieba.tokenize(test_sent)
        cutted = ' '.join(tk[0] for tk in result)
        return cutted


class Thulac(Cut):
    def __init__(self):
        super(Thulac, self).__init__(name='thulac')
        self.cut = thulac.thulac(seg_only=True)

    def cut_text(self, text):
        return self.cut.cut(text, text=True)


class Cleaner(object):
    def __init__(self, addtional_punc=''):
        self.all_punc = punctuation + '，。、【】“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^&#@￥×' + addtional_punc

    def __clean_tokens(self, text):
        tokens = text.split()
        tokens_clean = ' '.join([token for token in tokens if token not in self.all_punc])
        return tokens_clean

    def clean(self, train_text):
        return [self.__clean_tokens(doc) for doc in train_text]
