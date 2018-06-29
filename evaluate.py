from judger import Judger
import json
import os


class Evaluator(object):
    def __init__(self, predictor, input_path='./input', output='./out'):
        self.predictor = predictor
        self.input_path = input_path
        self.output_path = output
        self.judger = Judger('./data/accu.txt', './data/law.txt')
        self.cnt = 0

    def format_result(self, result):
        rex = {"accusation": [], "articles": [], "imprisonment": -3}

        res_acc = []
        for x in result["accusation"]:
            if not (x is None):
                res_acc.append(int(x))
        rex["accusation"] = res_acc

        if not (result["imprisonment"] is None):
            rex["imprisonment"] = int(result["imprisonment"])
        else:
            rex["imprisonment"] = -3

        res_art = []
        for x in result["articles"]:
            if not (x is None):
                res_art.append(int(x))
        rex["articles"] = res_art

        return rex

    def get_batch(self):
        v = self.predictor.batch_size
        if not (type(v) is int) or v <= 0:
            raise NotImplementedError

        return v

    def solve(self, fact):
        result = self.predictor.predict(fact)

        for a in range(0, len(result)):
            result[a] = self.format_result(result[a])

        return result

    def output_result(self, file_name):
        inf = open(os.path.join(self.input_path, file_name), "r")
        ouf = open(os.path.join(self.output_path, file_name), "w")

        fact = []

        for line in inf:
            fact.append(json.loads(line)["fact"])
            if len(fact) == self.get_batch():
                result = self.solve(fact)
                self.cnt += len(result)
                for x in result:
                    print(json.dumps(x), file=ouf)
                fact = []

        if len(fact) != 0:
            result = self.solve(fact)
            self.cnt += len(result)
            for x in result:
                print(json.dumps(x), file=ouf)
            fact = []

        ouf.close()

    def scoring(self, file_name):
        result = self.judger.test(self.input_path, self.output_path, file_name)
        return self.judger.get_score(result)

