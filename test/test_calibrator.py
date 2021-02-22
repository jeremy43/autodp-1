import sys
import os
from autodp.calibrator_zoo import eps_delta_calibrator,generalized_eps_delta_calibrator, ana_gaussian_calibrator
from autodp import rdp_bank
from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism,SubsampleGaussianMechanism, GaussianMechanism, ComposedGaussianMechanism, LaplaceMechanism
from autodp.transformer_zoo import Composition, AmplificationBySampling
from absl.testing import absltest
from absl.testing import parameterized

"""
Try calibrating noise to privacy budgets.
Cases 1: single parameter, no subsample or composition

"""

calibrate = eps_delta_calibrator()
ana_calibrate = ana_gaussian_calibrator()
eps = 0.1
delta = 1e-6
rel_tol = 0.01
mech1 = calibrate(ExactGaussianMechanism,eps,delta,[0,100],name='GM')
mech2 = ana_calibrate(ExactGaussianMechanism, eps, delta, name='Ana_GM')
print(mech1.name, mech1.params, mech1.get_approxDP(delta))
print(mech2.name, mech2.params, mech2.get_approxDP(delta))
params = ({'eps':0.1, 'delta':1e-6, 'result':0.1},{'eps':0.2, 'delta':1e-6, 'result':0.2})

def _caliborator_single(cur_eps,cur_delta):
    mech1 = calibrate(ExactGaussianMechanism, cur_eps, cur_delta, [0, 100], name='GM')
    return mech1.get_approxDP(delta)


class TestDPSGDPoisson(parameterized.TestCase):

    @parameterized.parameters(p for p in params)
    def test_RDP_values_lowerbound(self, eps, delta, result):
        cali_res = _caliborator_single(eps,delta)
        self.assertGreaterEqual(cali_res, (1. - rel_tol) * result)



"""

Cases 2: Test calibration with Gaussian under composition, coeff is the number of composition
We now have multiple parameters --- params['coeff'] and params['sigma']. 
The coeff is fixed and the calibrator optimize over sigma. We use para_name to denote the parameter that we want to optimize.

"""
coeff = 20
general_calibrate = generalized_eps_delta_calibrator()
params = {}
params['sigma'] = None
params['coeff'] = 20

#mech3 = general_calibrate(ComposedGaussianMechanism, eps, delta, [0,1000],params=params,para_name='sigma', name='Composed_Gaussian')
#print(mech3.name, mech3.params, mech3.get_approxDP(delta))

"""
Cases 3: Test calibration with SubsampledGaussian 
We now have three parameters --- params['coeff'], params['prob'] and params['sigma']. 
The coeff and prob are fixed and the calibrator optimize over sigma. We use para_name to denote the parameter that we want to optimize.

"""
params['prob'] = 0.01
#mech4 = general_calibrate(SubsampleGaussianMechanism, eps, delta, [0,1000],params=params,para_name='sigma', name='Subsampled_Gaussian')
#print(mech4.name, mech4.params, mech4.get_approxDP(delta))




"""

Cases 4: single parameter for Laplace mechanism, no subsample or composition

"""

calibrate = generalized_eps_delta_calibrator()

eps = 0.1
delta = 1e-6
#mech = calibrate(LaplaceMechanism,eps,delta,[0,100],name='Laplace')
#print(mech.name, mech.params, mech.get_approxDP(delta))

if __name__ == '__main__':
    absltest.main()
