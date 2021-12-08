#------------------------------------------------
# Numerical solution to the diffusion equation
# in 2D using explicit finite differences
#------------------------------------------------
# written in python
# Accelerated with numba-cuda
#------------------------------------------------
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from numba import jit
from numba import cuda
from numba import *

@jit
def evolution(u,n_0,n_1,udx2_0,udx2_1,dt,kd,i):
    jp1 = i + n_0
    jm1 = i - n_0
    laplacian = (u[i-1]-2.0*u[i]+u[i+1])*udx2_0 + \
                 (u[jm1]-2.0*u[i]+u[jp1])*udx2_1
    out = u[i] + dt*kd*laplacian
    return out

evolution_gpu = cuda.jit(device=True)(evolution)

@cuda.jit
def solution_kernel(u_d,un_d,udx2_0,udx2_1,dt,n_0,n_1,kd):
   ii, jj = cuda.grid(2)
   i = ii + n_0*jj
   if ii==0 or jj==0 or ii==n_0-1 or jj==n_1-1: 
      out = 0.0
   else: 
      out = evolution_gpu(u_d,n_0,n_1,udx2_0,udx2_1,dt,kd,i)
   if i == int((n_0*n_1)/2)+int(n_0/2):
      out = 1.0
   un_d[i] = out

if __name__ == "__main__":
   #-------------------------------------------
   # number of cells
   n = np.array([512,512],dtype=np.int64)
   # Domain size ( less than one ) 
   L = np.array([1.0,1.0],dtype=np.float64)
   # diffusion constant 
   kd:float64 = 0.2
   # time steps 
   steps:int = 1000
   blockdim = (32,16)                                      
   #-------------------------------------------

   # Blocks
   griddim = (int(n[0]/blockdim[0]),int(n[1]/blockdim[1])) 
   # cell sizes
   dx = L/n
   udx2 = 1.0/(dx*dx)
   # time steps
   dt = 0.25*(min(dx[0],dx[1])**2)/kd
   print("dt = ",dt)
   # Total cells
   nt = n[0]*n[1]
   print("cells = ",nt)
   start = time.time()
   # initialize with zeros
   u  = np.zeros(nt,dtype=np.float64)  
   un = np.zeros(nt,dtype=np.float64)  
   # pass to device
   u_d = cuda.to_device(u) 
   un_d = cuda.to_device(un)
   # integrate in time
   for t in range(1,steps+1):
     solution_kernel[griddim,blockdim](u_d,un_d,udx2[0],udx2[1],dt,n[0],n[1],kd)
     u_d = cuda.to_device(un_d)
     if t%100==0: print("paso = ",t)
   # copy to cpu
   u_d.copy_to_host(u)
   end = time.time()
   print("Taken: ",end-start,"s")
   #---------------------------------------
   #-----------------
   # Plot
   #-----------------
   u = np.reshape(u,(n[0],n[1]))
   x,y = np.meshgrid(np.arange(0,L[0],dx[0]),np.arange(0,L[1],dx[1]))
   ax = plt.axes(projection='3d')
   ax.plot_surface(x,y,u,cmap=cm.hsv)
   plt.show()
