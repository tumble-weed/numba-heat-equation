#------------------------------------------------
# Numerical solution to the diffusion equation
# in 3D using explicit finite differences
#------------------------------------------------
# written in python
# Accelerated with numba-cuda
#------------------------------------------------
import numpy as np 
from mayavi import mlab
import time
from numba import jit
from numba import cuda
from numba import *

@jit
def evolution(u,n_0,n_1,n_2,udx2_0,udx2_1,udx2_2,dt,kd,i):
    jp1 = i + n_0
    jm1 = i - n_0
    kp1 = i + n_0*n_1
    km1 = i - n_0*n_1
    laplacian = (u[i-1]-2.0*u[i]+u[i+1])*udx2_0 + \
                 (u[jm1]-2.0*u[i]+u[jp1])*udx2_1 + \
                 (u[km1]-2.0*u[i]+u[kp1])*udx2_2
    out = u[i] + dt*kd*laplacian
    return out

evolution_gpu = cuda.jit(device=True)(evolution)

@cuda.jit
def solution_kernel(u_d,un_d,udx2_0,udx2_1,udx2_2,dt,n_0,n_1,n_2,kd):
   ii, jj , kk = cuda.grid(3)
   i = ii + n_0*jj + n_0*n_1*kk
   if ii==0 or jj==0 or kk==0 or ii==n_0-1 or jj==n_1-1 or kk==n_2-1: 
      out = 0.0
   else: 
      out = evolution_gpu(u_d,n_0,n_1,n_2,udx2_0,udx2_1,udx2_2,dt,kd,i)
   if i == int((n_0*n_1*n_2)/2)+int(n_0*n_1/2)+int(n_0/2):
      out = 1.0
   un_d[i] = out

if __name__ == "__main__":
   #-------------------------------------------
   # number of cells
   n = np.array([32,32,32],dtype=np.int64)
   # Domain size (less than one)
   L = np.array([1.0,1.0,1.0],dtype=np.float64)
   # diffusion constant
   kd:float64 = 0.2
   # time steps
   steps:int = 100000
   blockdim = (8,4,4)                                      
   #-------------------------------------------

   # blocks
   griddim = (int(n[0]/blockdim[0]),int(n[1]/blockdim[1]),int(n[2]/blockdim[2])) 
   # cell sizes
   dx = L/n
   udx2 = 1.0/(dx*dx)
   # time steps
   dt = 0.1*(min(dx[0],dx[1],dx[2])**2)/kd
   print("dt = ",dt)
   # total cells
   nt = n[0]*n[1]*n[2]
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
     solution_kernel[griddim,blockdim](u_d,un_d,udx2[0],udx2[1],udx2[2],dt,n[0],n[1],n[2],kd)
     u_d = cuda.to_device(un_d)
     if t%100==0: print("time = ",t)
   # copy to cpu
   u_d.copy_to_host(u)
   end = time.time()
   print("Taken: ",end-start,"s")
   #---------------------------------------
   #-----------------
   # plot
   #-----------------
   u = np.reshape(u,(n[0],n[1],n[2]))
   x,y,z = np.ogrid[0:L[0]:1j*n[0],0:L[1]:1j*n[1],0:L[2]:1j*n[2]]
   src = mlab.pipeline.scalar_field(u)
   mlab.pipeline.iso_surface(src, contours=[u.min()+0.1*u.ptp(), ],opacity=0.3)
   mlab.pipeline.iso_surface(src, contours=[u.max()-0.1*u.ptp(), ],)
   mlab.show()
