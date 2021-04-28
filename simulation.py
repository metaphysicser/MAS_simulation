# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time: 2021/4/15 16:51
# @USER: 86199
# @File: simulation
# @Software: PyCharm
# @Author: 张平路
------------------------------------------------- 
# @Attantion：
#    1、
#    2、
#    3、
-------------------------------------------------
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MAS import MAS
import tqdm
from tqdm import trange

import os, time, random

def simulation_pic3a(precision=200,round = 2000,save = True):
    x1 = np.linspace(0.8, 1, precision)
    x2 = np.linspace(1, 1.2, precision)
    x = np.append(x1, x2)
    y = np.empty(precision * 2)

    for i in trange(0, 2 * precision):
        m = MAS(s=x[i])
        y[i] = m.variance_func2(round)


    plt.yscale("log")
    plt.xlim(0.8, 1.2)
    plt.yticks([1e0, 1e5, 1e10])

    plt.grid()
    plt.xlabel('s')
    plt.ylabel('Empirical Var(θ∞)')
    plt.plot(x, y)
    if save:
       plt.savefig('figure/simulation1.jpg')


    plt.show()


def simulation_pic3b(precision=200, ave_n=200,save = True):
    x1 = np.linspace(0.8, 1, precision)
    x2 = np.linspace(1, 1.2, precision)

    x = np.append(x1, x2)
    y = np.empty((ave_n, precision * 2))
    m = MAS(s=0.8)

    for j in trange(0, ave_n):
        for i in range(0, 2 * precision):
            m.update_s(x[i])
            y[j][i] = m.setting_time()
    y = y.mean(axis=0)

    plt.xlim(0.8, 1.2)
    plt.ylim(20,120)

    plt.figure(figsize=(10, 3))
    plt.grid()
    plt.xlabel('s')
    plt.ylabel('Ave. Setting Time')

    plt.plot(x, y)
    if save:
       plt.savefig('figure/simulation2.jpg')

    plt.show()

def simulation_pic4a(k = 100,repeat = 25,thresold = 0.01,save = True):


    convergence = 0

    ep = np.logspace(-2, 2, 21, base=10).tolist()
    x = []
    y = []

    for n in ep:
        e = n
        for j in trange(repeat):
            m = MAS(epsilon=e)
            m.q = np.zeros(m.number)
            for i in range(0, k):
                m.states_update()
                if m.convergence(thresold=thresold) and convergence == 0:
                    diff = abs(m.average_value() - m.initial_states.mean())

                    y.append(diff)
                    x.append(e)
                    convergence = 0
                    break



    plt.figure(figsize=(10, 3))
    plt.ylim(1e-4,1e2)

    plt.xlabel('$\overline{\epsilon}$')
    plt.ylabel('$θ∞-Ave(θ0)$')
    plt.xscale("log")
    plt.yscale("log")


    plt.grid()
    plt.scatter(x, y, s=80, facecolors='none', edgecolors='black')

    if save:
        plt.savefig('figure/simulation3.jpg')


    plt.show()


def simulation_no_noise(k=100, n=20):
    convergence = 0

    X = np.arange(0, k + 1)
    Y = []
    m = MAS(delta=0, number=n)

    Y.append(m.states.tolist())

    for i in trange(0, k):
        m.states_update()
        Y.append(m.states.tolist())
        if m.convergence(thresold=0.1) and convergence == 0:
            convergence = i

    if convergence == 0:
        convergence = k

    Y = np.array(Y).T
    for i in range(0, n):
        plt.plot(X, Y[i])

    plt.vlines(convergence, 0, 100)
    plt.savefig('figure/no_noise.jpg')
    plt.show()

    convergence = 0

    X = np.arange(0, k + 1)
    Y = []
    m.update_delta(1)


    Y.append(m.states.tolist())

    for i in trange(0, k):
        m.states_update()
        Y.append(m.states.tolist())
        if m.convergence(thresold=0.1) and convergence == 0:
            convergence = i

    if convergence == 0:
        convergence = k

    Y = np.array(Y).T
    for i in range(0, n):
        plt.plot(X, Y[i])

    plt.vlines(convergence, 0, 100)
    plt.savefig('figure/noise.jpg')
    plt.show()


def simulation_pic4b():


    ep = np.logspace(-2, 2, 21, base=10).tolist()
    x = []
    y = []

    y1 = []
    m = MAS(epsilon=0.01,alpha=0)
    for n in trange(len(ep)):
        m.update_epsilon(ep[n])
        x.append(ep[n])
        y.append(m.variance_func2())
        y1.append(m.variance_func3())

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('$\overline{\epsilon}$')
    ax.set_ylabel('Var(θ∞)')
    plt.plot(x, y1, color='red')
    plt.scatter(x, y, s=80, facecolors='none', edgecolors='black')
    plt.grid()
    plt.savefig('figure/simulation4.jpg')
    plt.show()

def simulation_pic5():
    m = MAS()
    #m.q = np.zeros(m.number)

    point = []
    for i in trange(1000):
        for j in range(10000):
            m.states_update()
            if m.convergence(thresold=0.01):
                point.append(m.average_value())
                break
        m.reset()

    print(point)

    plt.grid()
    fig, ax = plt.subplots()
    ax.set_xlabel('θ∞')
    plt.hist(point, bins=100, facecolor="blue", edgecolor="b", alpha=0.7)
    #plt.vlines(m.initial_states.mean(), 0, 300, colors='green')

    plt.savefig('figure/simulation5.jpg')
    plt.show()




if __name__ == "__main__":
    print("----Start----")
    simulation_pic4a()

















    print("----End------")
