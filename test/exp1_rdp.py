import numpy as np
from autodp.mechanism_zoo import GaussianMechanism, ExactGaussianMechanism, PureDP_Mechanism, LaplaceMechanism,SubSampleGaussian
import math
import matplotlib.font_manager as fm
from autodp.transformer_zoo import Composition, AmplificationBySampling
import os
import pickle
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

def exp2_gaussian():
    """
    rdp-base algorithm
    ours
    """
    sigma = 20

    delta = 1e-4

    k = 600 # the number of composition

    #klist = [1000]
    import pickle
    path = 'gaussian.pkl'
    #klist = [1]
    klist = [int(1.6**i) for i in range(int(math.floor(math.log(k,1.6)))+1)][6:]
    print('coeff list', klist)
    doc = {}
    for sigma in [4,10]:
        eps_rdp = []
        eps_phi = []
        eps_discrete = []
        for coeff in klist:
            rdp_gaussian = GaussianMechanism(sigma, name='Laplace')
            phi_gaussian= GaussianMechanism(sigma,coeff= coeff, CDF_off=False)
            discre_gaussian = GaussianMechanism(sigma,coeff= coeff, CDF_off=False)
            compose = Composition()
            composed_rdp_gaussian = compose([rdp_gaussian], [coeff])
            #eps_rdp.append(composed_rdp_gaussian.approxDP(delta))
            #eps_phi.append(phi_gaussian.approxDP(delta))
        print('in Gaussian with sigma',sigma)
        print('eps using phi', eps_phi)
        print('eps using rdp', eps_rdp)
        cur_result = {}
        cur_result['rdp'] = eps_rdp
        cur_result['eps'] = eps_phi
        doc[str(sigma)] = cur_result
    #with open(path, 'wb') as f:
    #    pickle.dump(doc, f)


    import matplotlib.pyplot as plt

    props = fm.FontProperties(family='Gill Sans', fname='/Library/Fonts/GillSans.ttc')
    f, ax = plt.subplots()
    plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(klist, doc['4']['rdp'], '-k', linewidth=2)
    plt.plot(klist, doc['4']['eps'], '-r^', linewidth=2)
    plt.plot(klist, doc['10']['rdp'], '--k', linewidth=2)
    plt.plot(klist, doc['10']['eps'], '--r^', linewidth=2)
    plt.legend(
        [r'RDP with $\sigma=4$','$\phi$-function with $\sigma=4$','RDP with $\sigma=10$','$\phi$-function with $\sigma=10$'], loc='best', fontsize=17)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'Number of Composition', fontsize=20)
    plt.ylabel(r'$\epsilon$', fontsize=20)
    ax.set_title('Title', fontproperties=props)
    #plt.show()
    plt.savefig('exp1_gaussian.pdf', bbox_inches='tight')


#exp2_gaussian()
def exp2_laplace():
    """
    rdp-base algorithm
    ours
    """
    b = 1

    delta = 1e-4

    k = 600 # the number of composition
    eps_rdp =[]
    eps_phi = []
    ##klist = [1000]
    import pickle
    path = 'laplace.pkl'
    klist = [1]
    klist = [int(1.6**i) for i in range(int(math.floor(math.log(k,1.6)))+1)][6:]

    print('coeff list', klist)
    doc = {}

    if not os.path.exists(path):
        for b in [10, 20]:
            eps_rdp = []
            eps_phi = []
            for coeff in klist:
                rdp_laplace = LaplaceMechanism(b, name='Laplace')
                phi_laplace= LaplaceMechanism(b,coeff= coeff, CDF_off=False)
                compose = Composition()
                composed_rdp_laplace = compose([rdp_laplace], [coeff])
                eps_rdp.append(composed_rdp_laplace.approxDP(delta))
                eps_phi.append(phi_laplace.approxDP(delta))
            print('in Laplace with lambda',b)
            print('eps using phi', eps_phi)
            print('eps using rdp', eps_rdp)
            cur_result = {}
            cur_result['rdp'] = eps_rdp
            cur_result['eps'] = eps_phi
            doc[str(b)] = cur_result
        with open(path, 'wb') as f:
            pickle.dump(doc, f)
    else:
        with open(path,'rb') as f:
            doc =pickle.load(f)
            klist =klist[:]

    import matplotlib.pyplot as plt

    props = fm.FontProperties(family='Gill Sans', fname='/Library/Fonts/GillSans.ttc')
    f, ax = plt.subplots()
    plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(klist, doc['10']['rdp'], '-k', linewidth=2)
    plt.plot(klist, doc['10']['eps'], '-r^', linewidth=2)
    plt.plot(klist, doc['20']['rdp'], '--k', linewidth=2)
    plt.plot(klist, doc['20']['eps'], '--r^', linewidth=2)
    plt.legend(
        [r'RDP with $\lambda=10$','$\phi$-function with $\lambda=10$','RDP with $\lambda=20$','$\phi$-function with $\lambda=20$'], loc='best', fontsize=17)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'Number of Composition', fontsize=20)
    plt.ylabel(r'$\epsilon$', fontsize=20)
    ax.set_title('Title', fontproperties=props)
    #plt.show()
    plt.savefig('exp1_laplace.pdf', bbox_inches='tight')


#exp2_laplace()

#exp2_gaussian()
def exp3_subsample():
    """
    rdp-base algorithm
    ours
    """

    delta = 1e-5
    k = 10000 # the number of composition
    eps_rdp =[]
    eps_phi = []
    ##klist = [1000]
    import pickle
    prob = 0.01
    path = 'subsample_sigma_1_1.pkl'
    #klist = [10]
    klist = [int(1.6**i) for i in range(int(math.floor(math.log(k,1.6)))+1)][4:]
    klist = klist[3:3]
    #klist = [ 1844, 2951, 472, 7555]
    #rdp[1.0751861799408282, 1.0930028001306573, 1.1215093936893195,1.24088, ]
    #eps_phi = [0.32307504219142436, 0.5178701140088057, 0.5549947513146491, 0.63736,0.738338, 0.8688, 1.0403,1.2661,1.5641728579583056, 1.9570771269068095]
    print('coeff list', klist)
    doc = {}
    import os
    if not os.path.exists(path):
        for sigma in [1]:
            eps_rdp = []
            eps_phi = []
            for coeff in klist:
                gm1 = ExactGaussianMechanism(sigma, name='GM1')
                compose = Composition()
                poisson_sample = AmplificationBySampling(PoissonSampling=True)
                composed_mech = compose([poisson_sample(gm1, prob, improved_bound_flag=True)], [coeff])
                phi_subsample = SubSampleGaussian(sigma, prob,coeff, CDF_off=False)
                eps_rdp.append(composed_mech.approxDP(delta))
                eps_phi.append(phi_subsample.approxDP(delta))

                print('eps using phi', eps_phi)
                print('eps using rdp', eps_rdp)
            cur_result = {}
            cur_result['rdp'] = eps_rdp
            # the following is for sigma =1 delta= 1e-5，coeff = [6, 10, 16]
            #eps_phi = [0.35, 0.3902，0.4369]
            cur_result['eps'] = eps_phi
            #cur_result['eps'] = [0.32307504219142436, 0.5178701140088057, 0.5549947513146491, 0.63736,0.738338, 0.8688, 1.0403,1.2661,1.5641728579583056, 1.9570771269068,2.472411923202513, 3.134941947019083, 3.93493886010072, 4.812796370665863]
            #eps_phi
            #the following is for sigma = 2
            #eps_phi=[0.32090208025649297, 0.6933024857345125, 0.934774041513957, 0.23758985302729116, 0.29877869729288303, 0.37769860329198085, 0.47874302540094793, 0.6085529670774328, 0.7757476168763454, 0.9915544835359067,1.269970297648549, 1.6288395714082657, 2.0872368031842625]
            #eps_rdp=[ 0.27944560876961533, 0.288844914692552, 0.3041187790179954, 0.3282044952714811, 0.36697661565260564, 0.42918374829367234, 0.523247464337445, 0.6621510760685007, 0.8418952164858443, 1.073359779561713,1.3728847499713157, 1.760174239669424, 2.2605628424596373]
            doc[str(sigma)] = cur_result
        with open(path, 'wb') as f:
            pickle.dump(doc, f)
    else:
        with open(path,'rb') as f:
            doc =pickle.load(f)
            klist =klist[:]

    import matplotlib.pyplot as plt

    props = fm.FontProperties(family='Gill Sans', fname='/Library/Fonts/GillSans.ttc')
    f, ax = plt.subplots()
    plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.loglog(klist[1:], doc['1']['rdp'][1:], '-k', linewidth=2)
    plt.loglog(klist[1:], doc['1']['eps'][1:], '-r^', linewidth=2)
    #plt.plot(klist, doc['20']['rdp'], '--k', linewidth=2)
    #plt.plot(klist, doc['20']['eps'], '--r^', linewidth=2)
    plt.legend(
        [r'RDP with $\sigma=1,\gamma = 1e-2$','$\phi$-function with $\sigma=1,\gamma = 1e-2$'], loc='best', fontsize=17)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.ylim([0.3, 10])
    plt.yticks(fontsize=20)
    plt.xlabel(r'Number of Composition', fontsize=20)
    plt.ylabel(r'$\epsilon$', fontsize=20)
    ax.set_title('Title', fontproperties=props)
    #plt.show()
    plt.savefig('exp3_subsampling.pdf', bbox_inches='tight')

#exp3_subsample()



def exp4_subsample_fixed_eps():
    """
    x axis is # composition
    y axis is the fixed privacy budget
    ours
    """

    eps = 1
    k = 10000 # the number of composition
    eps_rdp =[]
    eps_phi = []
    ##klist = [1000]
    import pickle
    prob = 0.02

    #klist = [10]
    klist = [100* i for i in range(2,10)]

    print('coeff list', klist)
    doc = {}
    exp4_path = 'gamma_0.02_sigma_2.pkl'
    if os.path.exists(exp4_path):
        with open(exp4_path, 'rb') as f:
            doc = pickle.load(f)
            klist = klist[:]
            eps_phi = doc['phi']
            eps_rdp = doc['rdp']
    else:

        for sigma in [2]:
            eps_rdp = []
            eps_phi = []
            for coeff in klist:
                gm1 = ExactGaussianMechanism(sigma, name='GM1')
                compose = Composition()
                poisson_sample = AmplificationBySampling(PoissonSampling=True)
                composed_mech = compose([poisson_sample(gm1, prob, improved_bound_flag=True)], [coeff])
                #phi_subsample = SubSampleGaussian(sigma, prob,coeff, CDF_off=False)
                eps_rdp.append(composed_mech.approx_delta(eps))
                #eps_phi.append(phi_subsample.approx_delta(eps))

                print('eps using phi', eps_phi)
                print('eps using rdp', eps_rdp)
            cur_result = {}
            cur_result['rdp'] = eps_rdp
            # the following is for sigma =1 delta= 1e-5，coeff = [6, 10, 16]
            eps_phi = [8.126706321817492e-11, 1.680813696457811e-08, 3.7171456678093376e-07, 2.8469414374309687e-06, 1.2134517188197369e-05, 3.612046366420398e-05, 8.487142126792672e-05, 0.00016916885963894502]

            cur_result['phi'] = eps_phi

            with open(exp4_path, 'wb') as f:
                pickle.dump(cur_result, f)




    import matplotlib.pyplot as plt
    #epsusingphi[1.7183697391746319e-06, 1.735096619083328e-06, 2.090003729609751e-06]
    #epsusingrdp[1.1702248059464182e-09, 4.4203574134371593e-10, 3.824438543631459e-10]
    props = fm.FontProperties(family='Gill Sans', fname='/Library/Fonts/GillSans.ttc')
    f, ax = plt.subplots()
    plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(klist,eps_rdp, '-k', linewidth=2)
    plt.plot(klist, eps_phi, '-r^', linewidth=2)
    plt.yscale('log')
    #plt.plot(klist, doc['20']['rdp'], '--k', linewidth=2)
    #plt.plot(klist, doc['20']['eps'], '--r^', linewidth=2)
    plt.legend(
        [r'RDP with $\sigma=2,\gamma = 2e-2$','$\phi$-function with $\sigma=2,\gamma = 2e-2$'], loc='best', fontsize=17)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'Number of Composition', fontsize=20)
    plt.ylabel(r'$\delta$', fontsize=20)
    ax.set_title('Title', fontproperties=props)
    #plt.show()
    plt.savefig('exp4_delta.pdf', bbox_inches='tight')

exp4_subsample_fixed_eps()
