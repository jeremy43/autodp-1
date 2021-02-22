
import numpy as np
import math
import pickle
from autodp import utils,rdp_bank,phi_bank
from scipy import special
from scipy.stats import norm
import scipy.integrate as integrate

from scipy.optimize import minimize_scalar


def _log1mexp(x):
    """ from pate Numerically stable computation of log(1-exp(x))."""
    if x < -1:
        return math.log1p(-math.exp(x))
    elif x < 0:
        return math.log(-math.expm1(x))
    elif x == 0:
        return -np.inf
    else:
        raise ValueError("Argument must be non-positive.")


def stable_log_diff_exp(x):
    # ensure that y > x
    # this function returns the stable version of log(exp(y)-exp(x)) if y > x

    mag = np.log(1 - np.exp(x - 0))

    return mag


def cdf_approx(phi, ell, L=100.,n=100000):
    """
    L shall not choose it to be too small
    this code is used to evaluate CDF when there is no closed form expression
    :param phi: the characteristic function

    ell: return the CDF when the privacy loss RV is evaluated at ell
    L: limit of the integral
    n: number of discretisation points
    :return: Evaluation of the RDP's epsilon
    """

    #cdf = integrate.quad(lambda x: (np.exp(x*1.j*L)-np.exp(-x * ell * 1.0j))/(1.0j*x) * np.exp(phi(x)), -L, -0.000000001)[0] + \
    #      integrate.quad(lambda x: (np.exp(x*1.j*L)-np.exp(-x * ell * 1.0j))/(1.0j*x) * np.exp(phi(x)), 0.000000001, L)[0]
    #print('whether pi = ',integrate.quad(lambda x: (np.exp(x*1.j*L))/(1.0j*x) * np.exp(phi(x)), -L, -0.000000001)[0] + \
    #      integrate.quad(lambda x: (np.exp(x*1.j*L))/(1.0j*x) * np.exp(phi(x)), 0.000000001, L)[0])
    # the following based on integrals over infinite interval
    #f(t/(1-t**2)*(1+t**2)/(1-t**2)**2 where t in [-1, 1]
    #cdf = integrate.quad(lambda t: (- np.exp(-(t/(1-t**2)) * ell * 1.0j)) / (1.0j * (t/(1-t**2))) * np.exp(phi(t/(1-t**2)))*(1.+t**2)/((1-t**2)**2), -1+1e-3,
    #                   1-1e-3)[0]
    #cdf = max(cdf/(2*np.pi),0)
    #print('scify integration cdf',cdf)

    # the following use riemann sum to approximate levy theorem
    #N = 200000
    N = 5000000
    """
    dx = 2.0 * L / N  # discretisation interval \Delta x
    t = np.linspace(-L, L - dx, N, dtype=np.complex128)
    t[int(N/2)] = t[int(N/2)+1]
    phi_result = [phi(each_t) for each_t in t]
    phi_result = np.array(phi_result, dtype=np.complex128)
    term_1 = np.log(1.0j/t) -1.j*t*ell +phi_result
    log_result = utils.stable_logsumexp(term_1)
    riemann_result = np.real(np.exp(log_result)*dx/(2*np.pi) +0.5)
    print('ell',ell,'riemann cdf', riemann_result,  min(max(riemann_result,0),1))
    #return  min(max(riemann_result,0),1)
    """
    #return riemann_result


    # the following use quadrature
    def qua(t):
        #print('integrade t', len(t)) #max iter indicates how many sampled points
        new_t = t*1.0/(1-t**2)
        phi_result = [phi(x) for x in new_t]
        phi_result = np.array(phi_result, dtype=np.complex128)
        inte_function = 1.j/new_t * np.exp(-1.j*new_t*ell)*np.exp(phi_result)
        return inte_function
    inte_f = lambda t: qua(t)*(1+t**2)/((1-t**2)**2)
    from scipy import integrate
    res = integrate.quadrature(inte_f, -1.0, 1.0, maxiter=1000)
    result = res[0]
    error = res[1]
    print('quadrature result', np.real(result)/(2*np.pi)+0.5, 'error range', error)
    return np.real(result)/(2*np.pi)+0.5
    # the following use riemann sum integrals over infinite intervals
    """
    ddx = 2.0/N
    
    t = np.linspace(-1+ddx, 1 - ddx, N, dtype=np.complex128)
    new_t = t/(1-t**2)
    save_phi_path = str(N)+'phi.pkl'
    
    import os
    if os.path.exists(save_phi_path):
        with open(save_phi_path, 'rb') as f:
            phi_result = pickle.load(f)
    else:
        phi_result = [phi(each_t) for each_t in new_t]
        with open(save_phi_path, 'wb') as f:
            
    

    # we now use log sum trick to solve it when coeff is large
    #term_1 =np.log(1.j/new_t) -1.j * new_t * ell+phi(new_t)
    ##term_3 = np.log((1+t**2))-2*np.log(1-t**2)
    #log_result = np.real(utils.stable_logsumexp(term_1+term_3))
    #cdf = np.real(np.exp(log_result) * ddx) / (2 * np.pi) + 0.5
    # not using log sum

    phi_result = [phi(each_t) for each_t in new_t]
    phi_result = np.array(phi_result, dtype=np.complex128)
    term_1 = np.log(1.0j/new_t) -1.j*new_t*ell +phi_result
    term_2 = np.log((1+t**2)/(1+t**4 - 2*t**2))
    print('before log sum')
    cdf = np.real(np.exp(utils.stable_logsumexp(term_1+term_2)))*ddx/(np.pi*2) +0.5
    print('integral based cdf', cdf)
    return cdf
    return min(max(riemann_result, 0), 1)
   
    """


"""

   ### the following is for the correct value
   sigma=20.0/(np.sqrt(1)) # simulate composition
   mean = 1.0 / (2.0 * sigma ** 2)
   std = 1.0 / (sigma)
   # there is a numerical issue when sigma>10, check borja's analytical gaussian method for numerical stable way
   correct_cdf = lambda x: norm.cdf((x - mean) / std)
   #print('g cdf',correct_cdf(ell))
   #return correct_cdf(ell)

   #print('correct cdf',correct_cdf(ell))
   return cdf
"""