# MLP with backpropagation learning of AND function
from pylab import *

# training vectors (Boolean AND function and constant input)
X=array([[0,0,1,1],[0,1,0,1],[1,1,1,1]])
Y=array([0,0,0,1])

# model specifications
Ni=3; Nh=3; No=1;

#parameter and array initialization
Ntrials=2000
h=zeros(Nh); y=zeros(No)
wh=randn(Nh,Ni); wo=randn(No,Nh) 
dwh=zeros(wh.shape); dwo=zeros(wo.shape) 
dh=zeros(Nh); do=zeros(No)  
error=zeros(Ntrials)

for trial in range(Ntrials):     
    #randomly pick training example
    pat=randint(4); x=X[:,pat] 
    
    #calculate prediction    
    h=1/(1+exp(-wh@x))
    y=1/(1+exp(-wo@h))

    # delta term for each layer (objective function error)   
    do=y*(1-y)*(Y[pat]-y)   
    dh=(h*(1-h))*(wo.transpose()@do)    
    
    # update weights with momentum
    dow=0.9*dwo+outer(h,do).T
    wo=wo+0.1*dwo
    dwh=0.9*dwh+outer(dh,x)
    wh=wh+0.1*dwh
    
    # test all pattern    
    h=1/(1+exp(-wh@X))
    y=1/(1+exp(-wo@h))   
    error[trial]=error[trial]+sum(abs(y-Y))

plot(error)
#savefig('tmp.eps', format='eps', dpi=1000)