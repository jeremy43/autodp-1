import numpy as np
from autodp.mechanism_zoo import GaussianMechanism, ExactGaussianMechanism, PureDP_Mechanism, LaplaceMechanism,SubSampleGaussian
import math
import matplotlib.font_manager as fm
from autodp.transformer_zoo import Composition, AmplificationBySampling

def exp_1_fdp():
    # Example 1: Gaussian mechanism
    sigma = 2.0


    gm0 = GaussianMechanism(sigma,name='GM0',approxDP_off=True, use_basic_RDP_to_approxDP_conversion=True)
    gm1 = GaussianMechanism(sigma,name='GM1',approxDP_off=True)
    gm1b = GaussianMechanism(sigma,name='GM1b',approxDP_off=True, use_fDP_based_RDP_to_approxDP_conversion=True)
    gm2 = GaussianMechanism(sigma,name='GM2',RDP_off=True)
    gm3 = GaussianMechanism(sigma,name='GM3',RDP_off=True, approxDP_off=True, fdp_off=False)



    eps = np.sqrt(2)/sigma # Aligning the variance of the laplace mech and gaussian mech
    laplace = PureDP_Mechanism(eps,name='Laplace')

    label_list = ['Naive_RDP_conversion','Balle et al., 2020','Our improved conversion',
                  'Exact_eps_delta_DP','Exact_fdp']


    import matplotlib.pyplot as plt



    fpr_list, fnr_list = gm0.plot_fDP()
    fpr_list1, fnr_list1 = gm1.plot_fDP()
    fpr_list1b, fnr_list1b = gm1b.plot_fDP()
    fpr_list2, fnr_list2 = gm2.plot_fDP()
    fpr_list3, fnr_list3 = gm3.plot_fDP()
    #fpr_list4, fnr_list4 = laplace.plot_fDP()

    plt.figure(figsize=(4,4))
    plt.plot(fpr_list,fnr_list)
    plt.plot(fpr_list1,fnr_list1)
    plt.plot(fpr_list1b,fnr_list1b)
    plt.plot(fpr_list2, fnr_list2)
    plt.plot(fpr_list3, fnr_list3,':')
    #plt.plot(fpr_list4, fnr_list4,'-.')
    plt.legend(label_list)
    plt.xlabel('Type I error')
    plt.ylabel('Type II error')
    plt.savefig('rdp2fdp.pdf')
    plt.show()

exp_1_fdp()