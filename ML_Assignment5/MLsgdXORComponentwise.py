# MLP with backpropagation learning of AND function
import numpy as np
import matplotlib.pyplot as plt

# training vectors (Boolean AND function and constant input)
X=np.array([[0,0,1,1],[0,1,0,1],[1,1,1,1]])
Y=np.array([0,0,0,1])

# model specifications
Ni=3; Nh=3; No=1;

#parameter and array initialization
Ntrials=2000
h=np.zeros(Nh); y=np.zeros(No)
wh=np.random.randn(Nh, Ni); wo=np.random.randn(No, Nh) 
dwh=np.zeros(wh.shape); dwo=np.zeros(wo.shape) 
dh=np.zeros(Nh); do=np.zeros(No)  
error=np.zeros(Ntrials)

for trial in range(Ntrials): 
    #pick example pattern randomly
    pat=np.random.randint(4); x=X[:,pat]
    
    #calculate hidden states of sigmoid nodes
    for ih in range(Nh):  #for each hidden node
        sumInput=0
        for ii in range(Ni):  #loop over input features
            sumInput=sumInput+wh[ih,ii]*x[ii]
        h[ih]=1/(1+np.exp(-sumInput))
    #calclate output (prediction) with sigm=oid nodes
    for io in range(No):  #for each output node
        sumInput=0;
        for ih in range(Nh):  #loop over inputs from hidden
            sumInput=sumInput+wo[io,ih]*h[ih];
        y[io]=1/(1+np.exp(-sumInput))
    
    # delta term for each layer (objective function error)    
    for io in range(No):  #for each output node
        do[io]=y[io]*(1-y[io])*(Y[pat]-y[io])    
    sumInput=0
    for ih in range(Nh):  #for each hidden node
        sumInput=0;   #backpropgation starts
        for io in range(No):
            sumInput=sumInput+wo[io,ih]*do[io]
        dh[ih]=h[ih]*(1-h[ih])*sumInput
    
    #update weights with momentum
    for io in range(No):
        for ih in range(Nh):
            dwo[io,ih]=0.9*dwo[io,ih]+h[ih]*do[io];
            wo[io,ih]=wo[io,ih]+0.1*dwo[io,ih];
    for ih in range(Nh):
        for ii in range(Ni):
            dwh[ih,ii]=0.9*dwh[ih,ii]+x[ii]*dh[ih];
            wh[ih,ii]=wh[ih,ii]+0.1*dwh[ih,ii];
 
    # test all pattern    
    for pat in range(4): 
        x=X[:,pat]
        #calculate prediction    
        for ih in range(Nh):  #for each hidden node
            sumInput=0
            for ii in range(Ni):  #loop over input features
                sumInput=sumInput+wh[ih,ii]*x[ii]
            h[ih]=1/(1+np.exp(-sumInput))
        for io in range(No):  #for each output node
            sumInput=0;
            for ih in range(Nh):  #loop over inputs from hidden
                sumInput=sumInput+wo[io,ih]*h[ih];
            y[io]=1/(1+np.exp(-sumInput))
            #y[io]=round(y[io])
            
        error[trial]=error[trial]+abs(y-Y[pat]);
plt.plot(error); 