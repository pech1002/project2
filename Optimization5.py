# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:36:13 2020

@author: pengxiang Cheng
"""
import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore',invalid='ignore')


class  optimization(object):
    def __init__(self,fun,gfun,hess,step,x_init):
        self.fun = fun
        self.gfun = gfun
        self.hess = hess
        self.step = step
        self.x_init = x_init   
        
    def Newton_class(self,x_new):
        
        gk = self.gfun(x_new)
        hk = self.hess(x_new)
        hk = (hk + hk)/2
        inverse = np.linalg.inv(hk) # Inverse hessian matrix
        s_k =  gk *inverse  # Newton Direction
        x_new = x_new -s_k # First Iteration
        
        return x_new
        
    def Newton(self):
        
        k = 1 # Record Interation times
        while k <= self.step:
            if k == 1:
                x_new = self.x_init
                x_new = self.Newton_class(x_new)
                k += 1
            else:
                x_new = self.Newton_class(x_new)
                k += 1      
        return x_new
    
    def Wolfe_powell(self,alfa,rho,sigma,gk,s_k,x0): #Wolfe-powell 
    
        m = 0 
        mk = 0
        while(m<20):
            
            if fun(x0 + alfa ** m * s_k) < (fun(x0) + rho*(alfa**m)*(gk*s_k.T))\
                and gfun(x0 + alfa**m *s_k)*s_k.T >= sigma * (gk * s_k.T):         
                mk = m
                break
            m += 1 
        return mk
    
    def Wolfe_inexact(self,alfa_0,alfa_L,gk,s_k,x0): #Wolfe-powell 
        m = 0
        while m < 20:
            
            gk1 = gfun(x0 + alfa_0*s_k)
            if fun(x0 + alfa_0*s_k) < fun(x0) + 0.01*alfa_0*(gk*s_k.T) and \
                  gfun(x0 + alfa_0 *s_k)*s_k.T >= 0.1* (gk * s_k.T):   #用Wolfe-powell 搜索步长
                  
                  ga_0 = gk1 * s_k.T
                  ga_L = gk * s_k.T
                  dela_alfa_0 = (alfa_0 - alfa_L)*ga_0/(ga_L-ga_0)
                  dela_alfa_0 = max(dela_alfa_0,0.1*(alfa_0 - alfa_L))
                  dela_alfa_0 = min(dela_alfa_0,9*(alfa_0 - alfa_L))
                  alfa_L = alfa_0
                  alfa_0 = alfa_0 + dela_alfa_0
                  break          
            m += 1
        return alfa_0
    
    def Coldstein(self,alfa,rho,sigma,gk,s_k,x0): #Goldstein
        m = 0 
        mk = 0
        while(m<20):
            # print("fun:",fun(x0 + alfa ** m * s_k))
            if fun(x0 + alfa ** m * s_k) < (fun(x0) + rho*(alfa**m)*(gk*s_k.T)) \
                  and fun(x0 + alfa ** m * s_k) > (fun(x0) + (1-rho)*(alfa**m)*(gk*s_k.T)):         
                mk = m
                break
            m += 1 
        
        return mk
    def DFP(self,Hk,sk,yk):
        Hk = Hk - (Hk * yk.T * yk * Hk) / (yk * Hk * yk.T) +\
                    (sk.T * sk) / (sk * yk.T) # DFP method
        return Hk
    
    def BFGS(self,Hk,sk,yk):
        
        yHy = (yk * Hk * yk.T) / (sk * yk.T)
        yHy = float(yHy)
        Hk = Hk + (1 + yHy) * ((sk.T * sk)/(sk * yk.T)) \
            - (sk.T*yk*Hk + Hk * yk.T*sk)/(sk * yk.T)   
        return Hk
    
    def Quasi_Newton(self):
        result = []
        x0 = self.x_init
        k_max =500
        alfa = 0.1
        rho = 0.1
        sigma = 0.7
        eps = 1e-5
        k = 0
        alfa_0 = 0.1
        alfa_L = 0.0
        # Hk = np.eye(2)
        Hk = np.linalg.inv(hess(x0))
        Hk = np.matrix(Hk)
        while(k < k_max):           
            gk = gfun(x0)
            s_k = -gk*Hk
            if np.linalg.norm(gk) < eps:
                break   
            # Wolfe and gold ,return mk alfa = alfa**mk
            #mk = self.Wolfe_powell(alfa,rho,sigma,gk,s_k,x0)
            #mk = self.Coldstein(alfa,rho,sigma,gk,s_k,x0)
            # x = x0 +alfa**mk *s_k
            
            # Wolfe inexact  return alfa_0
            alfa_0 = self.Wolfe_inexact(alfa_0,alfa_L,gk,s_k,x0) #Wolfe-powell
            x = x0 +alfa_0 *s_k
            
            sk = x - x0
            yk = gfun(x) - gk  
            
            if sk * yk.T> 0:                  
                Hk = self.DFP(Hk,sk,yk)
                #Hk = self.BFGS(Hk,sk,yk)
            k += 1
            x0 = x
            result.append(self.fun(x0))
            
        return x0,result,k

    
if __name__ == '__main__' :
    
    def Contour_plot(levels):
     
        X1=np.arange(-0.5,2.0+0.05,0.05)
        X2=np.arange(-1.5,4+0.05,0.05)
        [x1,x2]=np.meshgrid(X1,X2)
        f = (1-x1)**2 +100*(x2 - x1**2)**2
        plt.figure()
        plt.contour(x1,x2,f,20,colors='k') # 画出函数的20条轮廓线
        plt.title("Rosenbrock Function:f(x,y)= (1-x)$^2$+100(y-x$^2$)$^2$") 
        plt.xlabel("x")
        plt.ylabel("y")
        
        plt.figure()
        contour = plt.contour(x1,x2,f,levels = [0.02,1],colors='k')
        plt.clabel(contour,fontsize=10,colors=('k','r'))
        plt.title("Steps of Powell's method to computer a minimum") 
        plt.xlabel("x")
        plt.ylabel("y")
    
        
    #fun  function
    #gfun gradient matrix
    #hess Hessian matrix
    fun =  lambda x:100*(x[0,1] - x[0,0]**2)**2 +(1 - x[0,0])**2
    gfun = lambda x:np.matrix([400*x[0,0]*(x[0,0]**2 - x[0,1]) + 2*(x[0,0] - 1),-200*(x[0,0]**2 - x[0,1])])
    hess = lambda x:np.matrix([[1200*x[0,0]**2 - 400*x[0,1] + 2,-400*x[0,0]],[-400*x[0,0],200]])
    x_init=np.matrix([1,0])
    opti = optimization(fun,gfun,hess,100,x_init)
    print("Newton:",opti.Newton())
    print("Quasi_newton:",opti.Quasi_Newton()[0]," Iteration:",opti.Quasi_Newton()[2],"Times")
    # levels = opti.Quasi_Newton()[1]
    # levels.sort()
    # Contour_plot(levels)
    
    
