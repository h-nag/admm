# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import numpy as np

max_iter = 100


def w_update():
    pass


def _prox_las(w, lambda_las):
    posi_index = w > lambda_las
    nega_index = w < lambda_las
    zero_index = abs(w) <= lambda_las

    prox = np.zeros(w.shape)
    prox[posi_index] = w[posi_index] - lambda_las
    prox[nega_index] = w[nega_index] + lambda_las
    prox[zero_index] = 0.0

    return prox

def _loss(x, w):
    L = -np.log(1 + np.exp(np.dot(w.T, x)))


def v_update():
    pass


def u_update():
    pass


def main(argv):
    rho = argv.p
    lambda_sen = argv.ls
    lambda_las = argv.ll
    w_vec = []
    D = 0
    for _ in range(max_iter):
        # w_vec = argmin(w, )
        for d in range(D):
            pass


def arg_parse():
    usage = "python %s [-p <Lagrangian variable>] [-ls <Strength for sen>] [-ll <Strength for las>]"

    arg = ArgumentParser(usage=usage)
    arg.add_argument('-p', dest='p', type=float, help="Augmented Lagrangian variable", requiredt=True)
    arg.add_argument('-ls', dest='ls', type=float, help="Regularization strength for sen", required=True)
    arg.add_argument('-ll', dest='ll', type=float, help="Regularization strength for las", required=True)

    return arg


if __name__ == '__main__':
    argv = arg_parse()
    main(argv)
