import numpy as np
import math
from autodp.mechanism_zoo import GaussianMechanism, PureDP_Mechanism
from sympy import *
from sympy.integrals.transforms import _fourier_transform
from sympy import fourier_transform,exp
import scipy.integrate as integrate
from scipy.fft import fft

#Consider function f(t)=1/(t^2+1)
#We want to compute the Fourier transform g(w)

#Discretize time t

t0=0.
dt=0.00001
end = 2*np.pi
t=np.arange(t0,end,dt)
#Define function
f=1./(t**2+1.)
N = end*100000
k = 20
w = 2*np.pi/(dt*N)
rieman = []
sym_result = integrate.quad(lambda y: np.exp(1.0j*k*y)*(1.0/(y**2+1)),0,end)
ne =[]
new_k = k/w
print('new_k',new_k)
for idx in range(len(t)):
    rieman.append(f[idx]*np.exp(-1.j*k*t[idx]))
    ne.append(f[idx]*np.exp(-1.j*new_k*w*t[idx]))
print('our with dw',sum(ne)*dt)
print('riemann', sum(rieman)*dt)
print('fourier result', sym_result)
#Compute Fourier transform by numpy's FFT function
g=fft(f).conj()
#frequency normalization factor is 2*np.pi/dt
#w = np.fft.fftfreq(f.size)*2*np.pi/dt

#In order to get a discretisation of the continuous Fourier transform
#we need to multiply g by a phase factor
new_k = int(k/w)
g*=dt*np.exp(-1.j*w*t0*new_k)
print(g[new_k])
"""
from scipy.fft import fft,ifft
x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
t0 = -100
dt = 0.001
N = 20000
index = 0
new_index = int((index-t0)/dt)
#index = int((4-t0)/dt)
t = np.arange(t0, -t0, dt)
f = 1./(t**2+1.)
w = 2.*np.pi/N
sym_result = integrate.quad(lambda y: np.exp(-1.0j*new_index*y)*(1.0/(y**2+1)),-100,100)
print('sym',sym_result)
phase = np.exp(-1.0j*index*2*np.pi/(N*dt)*t0)/(N)
y = fft(f)

print(y[new_index]*phase)
#y = Symbol('y')
t = 4
#t = Symbol('t')
#i = Symbol('i')
from sympy.abc import k,x

# f = 1/(y^2 +1)
#sym_result = integrate.quad(lambda y: np.exp(1.0j*t*y)*(1.0/(y**2+1)),-100,100)
#sym_result = integrate(exp(I*t*y)*1/(y**2+1),(y,-200,200))
#sym_four_result = _fourier_transform((1/(y**2+1)),y,t,1,-1,'hh')
#print('sum_result', sym_result, 'fourier_result',sym_four_result)



The following is using DFT method to approximate the characteristic function

t0 = -100
dt = 0.001
t = np.arange(t0, -t0, dt)
f = 1./(t**2+1.)
# compute using ny's fft function
g = np.fft.fft(f)

#In order to get a discretisation of the continuous Fourier transform
#we need to multiply g by a phase factor
#frequency normalization factor is 2*np.pi/dt
#w = 4*2*np.pi/(len(t))
w = np.fft.fftfreq(f.size)*2*np.pi/dt
#phase = np.exp(-1.j*4*len(f)/(2*math.pi)*t0)/(np.sqrt(2*np.pi))*dt
g*=dt*np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi))

print(math.pi*np.exp(-4))
#l = log(gamma**((2*y-1)*1.0/(2*sigma**2)) + 1-gamma)
#py = 1.0/(sqrt(2*pi*sigma**2))*(gamma*exp(-(y-1)**2/(2*sigma**2))+(1-gamma)*exp(-y**2/(2*sigma**2)))

#print(integrate(exp(-x**2),(x,1,2)))



# Example 1: Gaussian mechanism

sigma = 2.0
delta= 1e-4

gm0 = GaussianMechanism(sigma,name='GM0',approxDP_off=True, use_basic_RDP_to_approxDP_conversion=True)
gm1 = GaussianMechanism(sigma,name='GM1',approxDP_off=True)
gm1b = GaussianMechanism(sigma,name='GM1b',approxDP_off=True, use_fDP_based_RDP_to_approxDP_conversion=True)
gm2 = GaussianMechanism(sigma,name='GM2',RDP_off=True)
gm3 = GaussianMechanism(sigma,name='GM3',RDP_off=True, approxDP_off=True, fdp_off=False)

print('rdp',gm0.approxDP(delta),'exact_eps_delta',gm2.approxDP(delta),'fdp',gm3.approxDP(delta))

gm4 = GaussianMechanism(sigma,name='CDF',RDP_off=True, approxDP_off=True, fdp_off=True, CDF_off=False,log_off=True)
print('cdf',gm4.approxDP(delta))
eps = np.sqrt(2)/sigma # Aligning the variance of the laplace mech and gaussian mech
laplace = PureDP_Mechanism(eps,name='Laplace')

label_list = ['naive_RDP_conversion','BBGHS_RDP_conversion','Our new method',
              'exact_eps_delta_DP','exact_fdp',r'laplace mech ($b = \sqrt{2}/\sigma$)']


import matplotlib.pyplot as plt



fpr_list, fnr_list = gm0.plot_fDP()
fpr_list1, fnr_list1 = gm1.plot_fDP()
fpr_list1b, fnr_list1b = gm1b.plot_fDP()
fpr_list2, fnr_list2 = gm2.plot_fDP()
fpr_list3, fnr_list3 = gm3.plot_fDP()
fpr_list4, fnr_list4 = laplace.plot_fDP()

plt.figure(figsize=(4,4))
plt.plot(fpr_list,fnr_list)
plt.plot(fpr_list1,fnr_list1)
plt.plot(fpr_list1b,fnr_list1b)
plt.plot(fpr_list2, fnr_list2)
plt.plot(fpr_list3, fnr_list3,':')
plt.plot(fpr_list4, fnr_list4,'-.')
plt.legend(label_list)
plt.xlabel('Type I error')
plt.ylabel('Type II error')
plt.savefig('rdp2fdp.pdf')
plt.show()



delta = 1e-3


eps3 = gm3.approxDP(delta)
eps0 = gm0.approxDP(delta)
eps1 = gm1.approxDP(delta)
eps1b = gm1b.approxDP(delta)

eps2 = gm2.approxDP(delta)

eps4 = laplace.approxDP(delta)

epsilons = [eps0,eps1,eps1b,eps2,eps3,eps4]

print(epsilons)

plt.bar(label_list,epsilons)
plt.xticks(rotation=45, ha="right")
plt.show()

"""