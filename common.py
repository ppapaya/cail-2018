from abc import ABC, abstractmethod


class Nameable(ABC):
    def __init__(self, name):
        self.name = name


class ModelContext(object):
    def __init__(self, cut, vectorizer, model):
        self.cut = cut
        self.vectorizer = vectorizer
        self.model = model

    def cut_name(self):
        return self.cut.name

    def vectorizer_name(self):
        return '{}_{}'.format(self.cut.name, self.vectorizer.name)

    def model_name(self):
        return self.model.name

    def model_full_name(self):
        return '{}_{}_{}'.format(self.cut.name, self.vectorizer.name, self.model.name)
