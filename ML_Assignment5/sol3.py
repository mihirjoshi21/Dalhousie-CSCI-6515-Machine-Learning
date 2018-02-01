# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 02:45:51 2017

@author: mj
"""
import numpy as np

def binaryToAscii(value):
    inInt = int(value, 2)
    return inInt.to_bytes((inInt.bit_length() + 7) // 8, 'big').decode()

predict_p = {}    

totaltrials = 10

for count in range(totaltrials):
    data = np.loadtxt('pattern1')
        
    X = data.reshape(26, 156).T
    Y = []
    Y_val = np.arange(65, 91);
    for i in range(len(Y_val)):
        Y.append([int(x) for x in list('{0:0b}'.format(Y_val[i]))])
    Y = np.array(Y)    
    
    # model specifications
    Ni=156; Nh=82; No=7;
    
    aaa = []
    bbb = []
    #parameter and array initialization
    Ntrials=10000
    h=np.zeros(Nh); y=np.zeros(No)
    #wh=np.random.randn(Nh, Ni); wo=np.random.randn(No, Nh) 
    #aaa = wh.copy(); bbb = wo.copy();
    wh=np.loadtxt('wh.csv', delimiter=','); wo=np.loadtxt('wo.csv', delimiter=',');
    dwh=np.zeros(wh.shape); dwo=np.zeros(wo.shape) 
    dh=np.zeros(Nh); do=np.zeros(No)  
    
    for trial in range(Ntrials):     
        #randomly pick training example
        pat=np.random.randint(26); x=X[:,pat];
        
        #calculate prediction    
        h=1/(1+np.exp(-wh@x))
        y=1/(1+np.exp(-wo@h))
    
        # delta term for each layer (objective function error)   
        do=y*(1-y)*(Y[pat]-y)   
        dh=(h*(1-h))*(wo.transpose()@do)    
        
        # update weights with momentum
        dwo=0.7*dwo+np.outer(h,do).T
        wo=wo+0.1*dwo
        dwh=0.7*dwh+np.outer(dh,x)
        wh=wh+0.1*dwh
        
        # test all pattern    
        h=1/(1+np.exp(-wh@X))
        y=1/(1+np.exp(-wo@h))   
         
    
    pattern2 = np.loadtxt('pattern2')
    x_test = pattern2.ravel().T;
    h=1/(1+np.exp(-wh@x_test))
    y_predict=1/(1+np.exp(-wo@h))
    v = ""
    for j in y_predict:
        v += str(int(round(j)))
    
    letter = binaryToAscii(v);
    if(letter not in predict_p):
        predict_p[letter] = 1
    else:
        predict_p[letter] += 1
    
    print("Ran {0} out of {1} times".format(count+1, totaltrials))
        
print("3")
maximum = max(predict_p, key=predict_p.get)
print("Pattern is {0} as it is predicted {1} times out of {2}"
      .format(maximum, predict_p[maximum], totaltrials))