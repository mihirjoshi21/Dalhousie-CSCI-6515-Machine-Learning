# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 02:45:51 2017

@author: mj
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('pattern1')
    
X = data.reshape(26, 156).T
Y = []
Y_val = np.arange(65, 91);
for i in range(len(Y_val)):
    Y.append([int(x) for x in list('{0:0b}'.format(Y_val[i]))])
Y = np.array(Y)    

# model specifications
Ni=156; No=7; max_nodes = 400; number_of_exp = 20;

#parameter and array initialization
hiddn_nodes_array = np.arange(1, max_nodes)
errors_final = np.zeros(len(hiddn_nodes_array))
counter = 0
for hn in hiddn_nodes_array:
    Nh = hn
    whi=np.random.randn(Nh, Ni); woi=np.random.randn(No, Nh) 
    errors = np.zeros(number_of_exp)
    for b in range(number_of_exp):
        Ntrials=100
        h=np.zeros(Nh); y=np.zeros(No)
        wh = whi.copy(); wo = woi.copy();
        dwh=np.zeros(wh.shape); dwo=np.zeros(wo.shape) 
        dh=np.zeros(Nh); do=np.zeros(No)  
        error=np.zeros(Ntrials)
        
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
            error[trial]=error[trial]+np.sum(abs(y-Y.T))
        errors[b] = error[-1]
    errors_final[counter] = errors.min()
    counter += 1
    print("Error calculated for {0} hidden Nodes".format(Nh))

plt.figure(figsize=(15,15))
plt.plot(errors_final)
plt.xlabel('Number of Hidden Nodes')
plt.ylabel('Error')
plt.title('Error vs Hidden Nodes')
plt.show()