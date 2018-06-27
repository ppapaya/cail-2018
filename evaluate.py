from predictor import data
from sklearn.metrics import f1_score
import json
from judger import Judger


if __name__ == '__main__':
    judger = Judger('./data/accu.txt', './data/law.txt')
    result = judger.test('./input', './out', 'data_valid.json')
    score = judger.get_score(result)
    print(score)


