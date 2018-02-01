# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 02:45:51 2017

@author: mj
"""
import numpy as np
import matplotlib.pyplot as plt

def binaryToAscii(value):
    inInt = int(value, 2)
    return inInt.to_bytes((inInt.bit_length() + 7) // 8, 'big').decode()

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
Ntrials=2000
h=np.zeros(Nh); y=np.zeros(No)
#wh=np.random.randn(Nh, Ni); wo=np.random.randn(No, Nh) 
#aaa = wh.copy(); bbb = wo.copy();
wh=np.loadtxt('wh.csv', delimiter=','); wo=np.loadtxt('wo.csv', delimiter=',');
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
    error[trial]=error[trial]+sum(abs(y-Y.T))

plt.plot(error)
plt.show()
#savefig('tmp.eps', format='eps', dpi=1000)

# value of all predicted items
v= ""; w=""
count = 0
for i,k in zip(y.T,Y):
    v = ""; w=""
    for j,l in zip(i,k):
        v += str(int(round(j)))
        w += str(int(round(l)))
    if(v == w):
        count+=1
    print(binaryToAscii(v))

print("Accuracy Score is {0}".format(count/26*100))
     

# 2 
print("\n\n2")
bitsToFlip = np.arange(1,15)
bitsToFlip = bitsToFlip[1::2]

percent  = 0
checkLetter = '1000001'
for bitsFlip in bitsToFlip:
    percent = 0
    for i in range(1000):
        x_test = data[0:12].copy()
        fliping = np.random.randint(155, size=(bitsFlip))
        for i in fliping:
            x_test[int((i/12) - 1)][(i%13) - 1] = 1 if x_test[int((i/12) - 1)][(i%13) - 1]  == 0 else 0 
            
        #plt.imshow(x_test)
        x_test = x_test.ravel().T 
        hy=1/(1+np.exp(-wh@x_test))
        y_predict=1/(1+np.exp(-wo@hy))
        v = ""
        for j in y_predict:
            v += str(int(round(j)))
        if v == checkLetter:
            percent+=1
    print("Accuracy when {0} percent bits flipped 1000 times is {1}"
          .format(round((bitsFlip/156)*100), percent/10))