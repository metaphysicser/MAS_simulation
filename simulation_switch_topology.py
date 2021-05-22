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
from tqdm import trange,tqdm
from multiprocessing import Pool
import os, time, random
from Multi_threading import split_data


def simulation1(k=40, n=20, states = (0,10), r=1, delta=0, save=True):
    X = np.arange(0, k + 1)
    Y = []
    m = MAS_switch_topology(delta=delta, number=n, r=r,states = states)
    m.reset()
    m.r = r
    convergence = 0

    Y.append(m.states.tolist())
    for i in trange(0, k):
        m.states_update1()
        Y.append(m.states.tolist())
        if m.convergence():
            convergence = i
    Y = np.array(Y).T


    for i in range(0, n):
        plt.plot(X, Y[i])

    plt.vlines(convergence, states[0], states[1])
    if save:
        plt.savefig('figure3/simulation0')
    plt.show()


def simulation_settingtime(precision=21, delta=1, repeat=100, save=True):
    ep = np.logspace(-2, 2, precision, base=10).tolist()

    y1 = []
    y2 = []
    y3 = []
    y4 = []
    m = MAS_switch_topology(delta=delta)

    for e in ep:
        m.update_epsilon(e)
        time1 = 0
        time2 = 0
        time3 = 0
        time4 = 0
        for i in trange(repeat):
            time1 += m.setting_time(update=1)
            m.reset()
            time2 += m.setting_time(update=2)
            m.reset()
            time3 += m.setting_time(update=3)
            m.reset()
            time4 += m.setting_time(update=4,value=1/2)
            m.reset()
        y1.append(time1 / repeat)
        y2.append(time2 / repeat)
        y3.append(time3 / repeat)
        y4.append(time4 / repeat)

    plt.xscale("log")
    plt.grid()
    plt.xlabel('$\epsilon$')
    plt.ylabel('Setting time')
    plt.plot(ep, y1, label='no change')
    plt.plot(ep, y2, label='change to radius')
    plt.plot(ep, y3, label='change to radius/2')
    plt.plot(ep, y4, label='minius 1/2')
    plt.legend()

    if save:
        plt.savefig('figure3/simulation1')
    plt.show()

def simulation_var(delta = 1, repeat = 100,save = True):
    ep = np.logspace(-2, 2, 21, base=10).tolist()

    y1 = []
    y2 = []
    y3 = []
    m = MAS_switch_topology(delta=delta)
    for e in ep:
        m.update_epsilon(e)
        t1 = []
        t2 = []
        t3 = []

        for i in trange(repeat):

            m.update_mulity(update=1)
            base = m.states
            m.reset()

            m.update_mulity(update=2)
            var1 = m.states
            m.reset()

            m.update_mulity(update=3)
            var2 = m.states
            m.reset()

            var1 = np.mean(((var1 - base) ** 2))
            var2 = np.mean(((var2 - base) ** 2))

            t1.append(var1)
            t2.append(var2)

        y1.append(np.mean(t1))
        y2.append(np.mean(t2))




    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlabel('$\epsilon$')
    plt.ylabel('Var')
    plt.plot(ep, y1, label='resolution1')
    plt.plot(ep, y2, label='resolution2')

    plt.legend()

    if save:
        plt.savefig('figure3/simulation6')
    plt.show()
def simulation_var_var(delta = 1, repeat = 100,save = True):
    ep = np.logspace(-1, 1, 11, base=10).tolist()

    y1 = []
    y2 = []
    y3 = []
    m = MAS_switch_topology(delta=delta)
    for e in ep:
        m.update_epsilon(e)
        t1 = []
        t2 = []
        t3 = []

        for i in trange(repeat):

            m.update_mulity(update=1)
            t1.append(m.var())
            m.reset()

            m.update_mulity(update=2)
            t2.append(m.var())
            m.reset()

            m.update_mulity(update=3)
            t3.append(m.var())
            m.reset()

            # m.update_mulity(update=3)
            # var3 = m.states
            # m.reset()

            # var1 = np.mean(((var1 - base) ** 2))
            # var2 = np.mean(((var2 - base) ** 2))
            # var3 = np.mean(((var3 - base) ** 2))
            # t1.append(var1)
            # t2.append(var2)
            # t3.append(var3)
        y1.append(np.var(t1))
        y2.append(np.var(t2))
        y3.append(np.var(t3))



    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlabel('$\epsilon$')
    plt.ylabel('Var_var')
    plt.plot(ep, y1, label='no change')
    plt.plot(ep, y2, label='resolution1')
    plt.plot(ep, y3, label='resolution2')
    plt.legend()

    if save:
        plt.savefig('figure3/simulation5')
    plt.show()

def compute(repeat,result_list):
    m = MAS_switch_topology(delta=1)
    for i in range(repeat):
        m.update_mulity(update=1)
        result_list[0].append(m.var())
        m.reset()

        m.update_mulity(update=2)
        result_list[1].append(m.var())
        m.reset()




def simulation_var_avg(delta = 1, repeat = 100,save = True):
    ep = np.logspace(-1, 1, 11, base=10).tolist()


    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []
    m = MAS_switch_topology(delta=delta)
    for e in tqdm(ep):
        m.update_epsilon(e)
        t1 = []
        t2 = []
        t3 = []
        t4 = []
        t5 = []
        t6 = []

        order = np.arange(10)
        split_data(10,5,compute,[t1,t2])



        # for i in trange(repeat):
        #
        #     m.update_mulity(update=1)
        #     t1.append(m.var())
        #     m.reset()
        #
        #     m.update_mulity(update=2,value=1)
        #     t2.append(m.var())
        #     m.reset()
        #
        #     m.update_mulity(update=2,value=0)
        #     t3.append(m.var())
        #     m.reset()
        #
        #     m.update_mulity(update=2, value=0.5)
        #     t4.append(m.var())
        #     m.reset()
        #
        #     m.update_mulity(update=2, value=0.25)
        #     t5.append(m.var())
        #     m.reset()
        #
        #     m.update_mulity(update=2, value=0.75)
        #     t6.append(m.var())
        #     m.reset()
        #
        #
        y1.append(np.mean(t1))
        y2.append(np.mean(t2))
        y3.append(np.mean(t3))
        y4.append(np.mean(t4))
        y5.append(np.mean(t5))
        y6.append(np.mean(t6))



    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlabel('$\epsilon$')
    plt.ylabel('Var_avg')
    print(y1)
    print(y2)
    print(y3)
    print(y4)
    print(y5)
    print(y6)

    plt.plot(ep, y1, label='no change')
    plt.plot(ep, y2, label='arg 1')
    # plt.plot(ep, y3, label='arg 0')
    # plt.plot(ep, y4, label='arg 0.5')
    # plt.plot(ep, y5, label='arg 0.25')
    # plt.plot(ep, y6, label='arg 0.75')
    plt.legend()

    if save:
        plt.savefig('figure3/simulation3')
    plt.show()

def simulation_cluster_num(delta = 1, repeat = 100,save = True):
    ep = np.logspace(-2, 0, 41, base=10).tolist()

    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []
    m = MAS_switch_topology(delta=delta)
    for e in ep:
        m.update_epsilon(e)
        t1 = []
        t2 = []
        t3 = []
        t4 = []
        t5 = []
        t6 = []

        for i in trange(repeat):

            m.update_mulity(update=1)
            t1.append(m.cluster_num())
            m.reset()

            m.update_mulity(update=2,value=1)
            t2.append(m.cluster_num())
            m.reset()

            m.update_mulity(update=2,value = 0)
            t3.append(m.cluster_num())
            m.reset()

            m.update_mulity(update=2, value=0.5)
            t4.append(m.cluster_num())
            m.reset()

            m.update_mulity(update=2, value=0.25)
            t5.append(m.cluster_num())
            m.reset()

            m.update_mulity(update=2, value=0.75)
            t6.append(m.cluster_num())
            m.reset()

            # m.update_mulity(update=3)
            # var3 = m.states
            # m.reset()

            # var1 = np.mean(((var1 - base) ** 2))
            # var2 = np.mean(((var2 - base) ** 2))
            # var3 = np.mean(((var3 - base) ** 2))
            # t1.append(var1)
            # t2.append(var2)
            # t3.append(var3)
        y1.append(np.mean(t1))
        y2.append(np.mean(t2))
        y3.append(np.mean(t3))
        y4.append(np.mean(t4))
        y5.append(np.mean(t5))
        y6.append(np.mean(t6))



    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlabel('$\epsilon$')
    plt.ylabel('Cluster number')
    plt.plot(ep, y1, label='no change')
    plt.plot(ep, y2, label='arg 1')
    plt.plot(ep, y3, label='arg 0')
    plt.plot(ep, y4, label='arg 0.5')
    plt.plot(ep, y5, label='arg 0.25')
    plt.plot(ep, y6, label='arg 0.75')
    plt.legend()

    if save:
        plt.savefig('figure3/simulation4')
    plt.show()
if __name__ == "__main__":
    print("----Start----")

    # simulation1(k = 70,delta=0,n =30,states=(0,10))
    # simulation_cluster_num(repeat=10)
    simulation_var_avg(repeat=10)
    # simulation_var()
    # simulation_var_var()

    print("----End------")
