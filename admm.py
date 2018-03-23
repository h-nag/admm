# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import joblib

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

MODEL_NAME = ""
VOCAB_NAME = ""
CONTENTS_NAME = ""

max_iter = 100
top_r = 10


def w_update(lambda_las, step, rho, w, x, y):
    w_up = w - step * np.gradient(_loss(x, w, y) +)


def _prox_las(w, lambda_las):
    posi_index = w > lambda_las
    nega_index = w < lambda_las
    zero_index = abs(w) <= lambda_las

    prox = np.zeros(w.shape)
    prox[posi_index] = w[posi_index] - lambda_las
    prox[nega_index] = w[nega_index] + lambda_las
    prox[zero_index] = 0.0

    return prox


def _loss(x, w, y):
    L = np.log(1 + np.exp(-y * np.dot(w.T, x)))
    return L


def v_update():
    pass


def u_update(u, rho, v, M, w):
    u_new = u - rho * (v - np.dot(M, w))

    return u_new


def main(argv):
    # model load
    model = joblib.laod(argv.dir + MODEL_NAME)
    voc_model = joblib.laod(argv.dir + VOCAB_NAME)
    contents = joblib.load(argv.dir + CONTENTS_NAME)

    tp_word_ = model.topic_word_

    tp_list, voc = {}, []
    for tp_dist in tp_word_:
        tp_dict = {}
        topic_words = np.array(voc_model)[np.argsort(tp_dist)][:-(top_r + 1):-1]
        topic_dist = np.array(tp_dist)[np.argsort(tp_dist)][:-(top_r + 1):-1]
        voc.extend(topic_words)

        for i, word in enumerate(topic_words):
            tp_dict[word] = topic_dist[i]
        tp_list.append(tp_dict)

    voc = list(set(voc))
    topic_weight = {}
    w = []
    M = []
    for d in tp_dict:
        w_tmp = []
        for word in voc:
            if word in d:
                topic_weight[word] = d[word]
                w_tmp.append(d[word])
                voc_dist.append(d[word])
            else:
                topic_weight[word] = 0.0
                w_tmp.append(0.0)

        w.append(w_tmp)

    w = np.array(w)
    cv = CountVectorizer(vocabulary=voc)
    x = cv.fit_transform(contents)

    # M vector
    tp_num = tp_word_.shape[0]
    m_tmp = np.zeros(len(voc), tp_num)
    for i in enumerate(tp_num):
        v_array = np.where(w[i, :] > 0.0, 1, 0)

    rho = argv.p
    lambda_sen = argv.ls
    lambda_las = argv.ll
    w_vec = []
    u = np.zeros(v.shape)

    D = 0
    for _ in range(max_iter):
        # w_vec = argmin(w, )
        for d in range(D):
            pass


def arg_parse():
    usage = "python %s [-p <Lagrangian variable>] [-ls <Strength for sen>] [-ll <Strength for las>]" % __file__

    arg = ArgumentParser(usage=usage)
    arg.add_argument('-d', dest='dir', type=float, help="Directory path to model dump", requiredt=True)
    arg.add_argument('-p', dest='p', type=float, help="Augmented Lagrangian variable", requiredt=True)
    arg.add_argument('-ls', dest='ls', type=float, help="Regularization strength for sen", required=True)
    arg.add_argument('-ll', dest='ll', type=float, help="Regularization strength for las", required=True)
    arg.add_argument('-y', dest='y', type=int, help="Binary response given by feature vector and weight", default=1)
    arg.add_argument('-s', dest='step', type=float, help="Step width for proximal gradient method", default=0.5)

    return arg


if __name__ == '__main__':
    argv = arg_parse()
    main(argv)
