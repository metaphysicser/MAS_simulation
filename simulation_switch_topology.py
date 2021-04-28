# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time: 2021/4/16 11:31
# @USER: 86199
# @File: simulation_switch_topology
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
from MAS_switch_topology import MAS_switch_topology
import tqdm
from tqdm import trange
from multiprocessing import Pool
import os, time, random



def simulation_pic12(k = 40,n = 20,r = 2,delta = 0,save = True):

    X = np.arange(0, k + 1)
    Y = []
    m = MAS_switch_topology(delta=delta, number=n, r=r)
    m.initial_states =np.array([41.8309021  ,57.64950439 ,55.37732576 ,59.20001792, 57.7233624,  40.33882689,
 57.84884924, 48.73382393, 42.93908053 ,53.11672422, 56.06069219, 40.12895605,
 44.62407166, 47.55277699, 50.61738483, 40.01444508, 56.25899311, 49.84214835,
 50.72414305 ,41.86621181])
    m.reset()
    m.r = 4
    convergence = 0

    Y.append(m.states.tolist())
    for i in trange(0, k):
        m.states_update()
        Y.append(m.states.tolist())
        if m.convergence():
            convergence = i
    Y = np.array(Y).T
   # plt.title('r = 2')
    for i in range(0, n):
        plt.plot(X, Y[i])

    plt.vlines(convergence, 40, 60)



    if save:
        plt.savefig('figure2/simulation2b')

    plt.show()

def simulation_pic3b(precision=30, ave_n=200):
    x1 = np.linspace(0.8, 1, precision)
    x2 = np.linspace(1, 1.2, precision)

    x = np.append(x1, x2)
    y = np.empty((ave_n, precision * 2))
    m = MAS_switch_topology(s=0.8)

    for j in trange(0, ave_n):
        for i in trange(0, 2 * precision):
            m.update_s(x[i])
            y[j][i] = m.setting_time(thresold=1)
    y = y.mean(axis=0)

    fig, ax = plt.subplots()
    ax.set_xlim(0.8, 1.2)
    plt.plot(x, y)
    plt.grid()
    ax.set_xlabel('s')
    ax.set_ylabel('Ave. Setting Time')
    plt.savefig('figure2/simulation6.jpg')
    plt.show()


def simulation_pic4a(k = 10000,repeat = 25,save = True):


    convergence = 0

    ep = np.logspace(-2, 2, 21, base=10).tolist()
    x = []
    y = []

    for n in ep:
        e = n
        t = []
        for j in trange(repeat):
            m = MAS_switch_topology(epsilon=e)
            m.q = np.zeros(m.number)

            for i in range(0, k):
                m.states_update()
                if m.convergence(thresold=1) and convergence == 0:
                    y.append(m.variance_func())
                    x.append(e)
                    # t.append(m.variance_func())
                    convergence = 0
                    break
        # y.append(np.array(t).mean())
        # x.append(e)

    print(x,y)


    fig, ax = plt.subplots()
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel('$\overline{\epsilon}$')
    ax.set_ylabel('$Var(θ∞)$')
    plt.scatter(x, y, s=80, facecolors='none', edgecolors='black')
    plt.grid()
    if save:
        plt.savefig('figure2/simulation3.jpg')



    plt.show()


def simulation_pic4a2(k = 10000,repeat = 25):


    convergence = 0

    ep = np.logspace(-2, 2, 21, base=10).tolist()
    x = []
    y = []

    for n in ep:
        e = n
        t = []
        for j in trange(repeat):
            m = MAS_switch_topology(epsilon=e)
            m.q = np.zeros(m.number)

            for i in range(0, k):
                m.states_update()
                if m.convergence(thresold=0.1) and convergence == 0:
                    y.append(m.variance_func2())
                    x.append(e)
                    # t.append(m.variance_func())
                    convergence = 0
                    break
        # y.append(np.array(t).mean())
        # x.append(e)

    print(x,y)


    fig, ax = plt.subplots()
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel('$\overline{\epsilon}$')
    ax.set_ylabel('$entroy$')
    plt.scatter(x, y, s=80, facecolors='none', edgecolors='black')
    plt.grid()
    plt.savefig('figure2/simulation4.jpg')

    plt.show()

def simulation_pic4a3(k = 100000,repeat = 20):


    convergence = 0

    ep = np.logspace(-3, 3, 31, base=10).tolist()
    x = []
    y = []

    for n in ep:
        e = n
        t = []
        for j in trange(repeat):
            m = MAS_switch_topology(epsilon=e)
            m.q = np.zeros(m.number)

            for i in range(0, k):
                m.states_update()
                if m.convergence(thresold=0.001) and convergence == 0:
                    y.append(m.variance_func2()*m.variance_func())
                    x.append(e)
                    # t.append(m.variance_func())
                    convergence = 0
                    break
        # y.append(np.array(t).mean())
        # x.append(e)

    print(x,y)


    fig, ax = plt.subplots()
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel('$\overline{\epsilon}$')
    # ax.set_ylabel('$Var(θ∞)$')
    plt.scatter(x, y, s=80, facecolors='none', edgecolors='black')
    plt.grid()
    plt.savefig('figure2/simulation5.jpg')

    plt.show()

def simulation_7(precision=30, ave_n=200):


    ep = np.logspace(-3, 3, 31, base=10).tolist()


    y = np.empty((ave_n, len(ep)))
    m = MAS_switch_topology(s=1)

    for j in trange(0, ave_n):
        for i in trange(0, len(ep)):
            m.update_epsilon(ep[i])
            y[j][i] = m.setting_time(thresold=0.001)
    y = y.mean(axis=0)

    print(len(ep))
    print(y.shape)

    fig, ax = plt.subplots()
    ax.set_xscale("log")

    plt.plot(ep, y)
    plt.grid()
    ax.set_xlabel('$\overline{\epsilon}$')
    ax.set_ylabel('Ave. Setting Time')
    plt.savefig('figure2/simulation7.jpg')
    plt.show()


if __name__ == "__main__":
    print("----Start----")


    simulation_pic12(k =10,delta=0.1,save=False)



    print("----End------")
