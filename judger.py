from math import log
import os
import json


class Judger:
    # Initialize Judger, with the path of accusation list and law articles list
    def __init__(self, accusation_path, law_path):
        self.accu_dic = {}

        f = open(accusation_path, "r")
        self.task1_cnt = 0
        for line in f:
            self.task1_cnt += 1
            self.accu_dic[line[:-1]] = self.task1_cnt

        self.law_dic = {}
        f = open(law_path, "r")
        self.task2_cnt = 0
        for line in f:
            self.task2_cnt += 1
            self.law_dic[int(line[:-1])] = self.task2_cnt

    # Format the result generated by the Predictor class
    @staticmethod
    def format_result(result):
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

    # Gen new results according to the truth and users output
    def gen_new_result(self, result, truth, label):
        s1 = set(label["accusation"])
        s2 = set()
        for name in truth["accusation"]:
            s2.add(self.accu_dic[name.replace("[", "").replace("]", "")])

        for a in range(0, self.task1_cnt):
            in1 = (a + 1) in s1
            in2 = (a + 1) in s2
            if in1:
                if in2:
                    result[0][a]["TP"] += 1
                else:
                    result[0][a]["FP"] += 1
            else:
                if in2:
                    result[0][a]["FN"] += 1
                else:
                    result[0][a]["TN"] += 1

        s1 = set(label["articles"])
        s2 = set()
        for name in truth["relevant_articles"]:
            s2.add(self.law_dic[name])

        for a in range(0, self.task2_cnt):
            in1 = (a + 1) in s1
            in2 = (a + 1) in s2
            if in1:
                if in2:
                    result[1][a]["TP"] += 1
                else:
                    result[1][a]["FP"] += 1
            else:
                if in2:
                    result[1][a]["FN"] += 1
                else:
                    result[1][a]["TN"] += 1

        result[2]["cnt"] += 1
        sc = 0
        if truth["term_of_imprisonment"]["death_penalty"]:
            if label["imprisonment"] == -2:
                sc = 1
        elif truth["term_of_imprisonment"]["life_imprisonment"]:
            if label["imprisonment"] == -1:
                sc = 1
        else:
            if label["imprisonment"] < 0:
                sc = 0
            else:
                v1 = truth["term_of_imprisonment"]["imprisonment"]
                v2 = label["imprisonment"]
                v = abs(log(v1 + 1) - log(v2 + 1))
                if v <= 0.2:
                    sc = 1
                elif v <= 0.4:
                    sc = 0.8
                elif v <= 0.6:
                    sc = 0.6
                elif v <= 0.8:
                    sc = 0.4
                elif v <= 1.0:
                    sc = 0.2
                else:
                    sc = 0
        sc = sc * 1.0
        result[2]["score"] += sc

        return result

    # Calculate precision, recall and f1 value
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    @staticmethod
    def get_value(res):
        if res["TP"] == 0:
            if res["FP"] == 0 and res["FN"] == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
            recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    # Generate score for the first two subtasks
    def gen_score(self, arr):
        sumf = 0
        y = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for x in arr:
            p, r, f = self.get_value(x)
            sumf += f
            for z in x.keys():
                y[z] += x[z]

        _, __, f_ = self.get_value(y)

        return (f_ + sumf * 1.0 / len(arr)) / 2.0

    # Generatue all scores
    def get_score(self, result):
        s1 = self.gen_score(result[0])
        s2 = self.gen_score(result[1])
        s3 = 1.0 * result[2]["score"] / result[2]["cnt"]
        return [s1, s2, s3]

    # Test with ground truth path and the user's output path
    def test(self, truth_path, output_path, file_name):
        cnt = 0
        result = [[], [], {}]
        for a in range(0, self.task1_cnt):
            result[0].append({"TP": 0, "FP": 0, "TN": 0, "FN": 0})
        for a in range(0, self.task2_cnt):
            result[1].append({"TP": 0, "FP": 0, "TN": 0, "FN": 0})
        result[2] = {"cnt": 0, "score": 0}

        inf = open(os.path.join(truth_path, file_name), "r")
        ouf = open(os.path.join(output_path, file_name), "r")

        for line in inf:
            ground_truth = json.loads(line)["meta"]
            user_output = json.loads(ouf.readline())

            cnt += 1
            result = self.gen_new_result(result, ground_truth, user_output)

        return result
