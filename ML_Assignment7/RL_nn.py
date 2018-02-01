# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:47:35 2017

@author: mj
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


with tf.device('/gpu:0'):
    tf.reset_default_graph()
    
    inputs1 = tf.placeholder(shape=[1,5],dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([5,2],0,0.01))
    Qout = tf.matmul(inputs1,W)
    predict = tf.argmax(Qout,1)
    
    #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    nextQ = tf.placeholder(shape=[1,2],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

def calc_policy(Q):
    policy=np.zeros(5)
    for s in range(0,5):
        action_idx=np.argmax(Q[s,:])
        policy[s]=2*action_idx-1
        policy[0]=policy[4]=0
    return policy.astype(int)

def tau(s,a):
    if s==0 or s==4:  
        return(s)
    else:      
        l = a -1 if a[0]==0 else a;
        return(s + l)
        
def rho(s,a):
    return(s==1 and a==0)+2*(s==3 and a==1)  

# Set learning parameters
y = .5
e = 0.1
num_episodes = 5000
jList = []
targetQ = None
rList = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        s = 2
        rAll = 0
        d = False
        j = 0
        while j < 1000:
            j+=1
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(5)[s:s+1]})
            if np.random.rand() < e:
                a[0] = np.random.randint(2)
            r = rho(s, a)
            s1 = tau(s, a)[0]
            d = True if r>0 else False
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(5)[s1:s1+1]})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(5)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                if i%1000 == 0: 
                    print("episode completed {0}".format(i))
                e = 1./((i/50) + 10)
                break
    result = sess.run(W)
    policy = calc_policy(result)
    print('policy: \n',np.transpose(policy))
    print('Q values: \n',np.transpose(np.around(result, decimals=1)))