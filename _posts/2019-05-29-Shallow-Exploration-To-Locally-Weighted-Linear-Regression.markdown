---
    author: Kelvin
    #comments: true
    date: 2019-05-29
   layout: page
    title: Shallow Exploration To Locally Weighted Linear Regression(LWR)
    categories:
        - CS
        - MachineLearing
    tags:
        - Academic
---

# Writing Motivation
After a cursory study to the **Linear Regression**, I got some aerial view towards the algorithm. However recently I have reviewed this part in aspect of the mathematical derivation and code implementation,and discovered something new in conbinition of Statics knowledge. In view of my poor memory, I decide to record them in this low-level passage.

# Introduction
Regression is one kind of **Supervised learning** tasks, and what I am going to talk here is particularly Linear Regression ,which is a strategy in **Regression Learning** and attempt to establish a linear relationship between the inputset X and the predicted output $f(x)$. So what we want to achieve is to determine a model and find its parameters.

One fault in ordinary linear regression lies in its inclination to the under-fitting problem. It applies a hyperplane to fit the whole training set. But when the training set do not satisfy a Linear Distribution, in most cases the result of a linear regression model will generate a problem of under-fitting.

![](https://i.loli.net/2019/05/29/5cee862d5c15491757.png)

The **polynomial fit** is a very possible way towards the problem which can fit all of the training data. However, it behaves quite poor when it comes to predicting new samples because it causes the over-fitting problem and fail to fit the real model.

**Locally Weighted Linear Regression** performs as a rectification of the ordinary Linear Regression. To intuitively understand the concept, LWLR adds a new weight to the Loss Funciton $L(w,b)=\frac{1}{2}\sum_{i=1}^m\mu^{i}(f(x^{i})-y^{i})^2$.

Among them, **m** stants for the number of training samples.
$f(x)=\theta^Tx$, 
$\mu^i$ is a non-negative weights value, generally using an exponential form:$\mu^i=e^{-\frac{(x^i-x)^2}{2\tau^2}}$, it looks like **Normal Distribution**, 

Then we establish a Diagonal matrix $\psi$, s.t:
$$\psi_{i,i}=\mu^i=e^{-\frac{(x^i-x)^2}{2\tau^2}}$$
$x$ stands for the new input. 

$\tau$ stands for the extent to which the distant training data will be considered.
Smaller the value is ,deeper the fitting degree will be.

The Regularization Formula is $\theta = (X^T\psi X)^{-1}X^T\psi Y$ .
That is to say, every time we input new TEST Data, we will generate a new $\theta$ matrix.

We can intuitively understand the formula in this way: Smaller the distance between $x^i$ and the new $x$ is ,more significant the influence of the $i_{th}$ training data towards the result will be.

# Code Implement
```py
import numpy as np
import matplotlib.pyplot as plt

def LoadData(file):
        try:
                trainset=open(file)
                print("Openfile successfully.")
        except:
                print("openfile failure")
                return
        inset=[]
        labelset=[]
        for line in trainset.readlines():
                line=line.strip()
                LineArr=line.split('\t') #cut into list
                inset.append([float(LineArr[0]),float(LineArr[1])])
                labelset.append(float(LineArr[2]))
        return inset,labelset

def lwrtrain(testpoint,xarr,yarr,k=1.0):
        Xmat=np.mat(xarr)
        Ymat=np.mat(yarr)
        m=np.shape(Xmat)[0]
        weights = np.eye(m)
    
        for i in range(m):
                diffMat= testpoint-Xmat[i,:]
                weights[i,i]=np.exp(-(diffMat*(diffMat.T)/(k*k*2)))
        #xTx=np.zeros((np.shape(Xmat)[1]),(np.shape(Xmat)[1]))

        xTx=Xmat.T * (weights * Xmat ) #(2,2)=(2*n)*(n*n)*(n*2)
        if np.linalg.det(xTx)==0:
                print("This matrix cannot do inverse")
                return
        theta = xTx.I *(Xmat.T *(weights*Ymat.T))#(2*1)=(2*2)*(2*n)*(n*n)*(n*1)
        return testpoint*theta

def lwr(testArr,xarr,yarr,k=1.0):
        m=np.shape(testArr)[0]
        ytest=np.zeros(m)
        for i in range(m):
                ytest[i]=lwrtrain(testArr[i],xarr,yarr,k)
        return ytest

def show(xarr,yarr,ytest,k,a,fig):
        xMat=np.mat(xarr);yMat=np.mat(yarr)

        #对xarr排序
        strInd = xMat[:,1].argsort(0)#从小到大数据的序号
        xSort= np.sort(xMat,axis=0)
        #xSort= xMat[strInd]#[:,0,:]

        #Draw
        ax = fig.add_subplot(2,2,a)
        ax.plot(xSort,ytest[strInd])#line
        ax.scatter(xMat[:,1].flatten().getA() , yMat.flatten().getA())#point
        #only accept type:array, funciton flatten() return a matrix

        plt.title("k= %f" %k)
        #plt.show()

def run_main():
        xarr,yarr = LoadData("ex1.txt")
        print("LoadData Done.")

        Kk=[1.0,0.01,0.005,0.002]
        a=0
        fig=plt.figure()
        for k in Kk:
                a=a+1
                ytest=lwr(xarr,xarr,yarr,k)
                show(xarr,yarr,ytest,k,a,fig)
        plt.show()

if __name__== "__main__":
        run_main()

```
# Result Analysis
The result is presented below:
![](https://i.loli.net/2019/05/29/5cee9eff2b17273954.png)