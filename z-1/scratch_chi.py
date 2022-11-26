import emcee
import numpy as np
from scipy.odr import Model, RealData, ODR
import matplotlib as mpl
from matplotlib import pyplot as plt# import pyplot from matplotlib
import time               # use for timing functions
import corner

def straight_line(theta,x):
    y=theta[0]*x + theta[1]
    return y

min_ = -2
max_ = 12

def logprior(theta):
    lp = 0.
    m, c = theta
    lp = 0. if min_ < c < max_ and min_ < m < max_  else -np.inf
    # Gaussian prior on m
    mmu = 0.     # mean of the Gaussian prior
    msigma = 10. # standard deviation of the Gaussian prior
    lp -= 0.5 * ((m - mmu) / msigma)**2

    return lp

def logchi(theta, y, x, err_y):
    m,c=theta
    expected = m*x+c
    delta = y - expected
    chi_sq = np.sum(delta**2/err_y**2)
    # print(chi_sq)
    return -0.5*(chi_sq)

def logposterior(theta, y, x, err_y, err_x):
    lp = logprior(theta) # get the prior
    # if the prior is not finite return a probability of zero (log probability of -inf)
    if not np.isfinite(lp):
        return -np.inf

    # return the likeihood times the prior (log likelihood plus the log prior)
    # return lp + lnprob_vertical(theta, x, err_x, y, err_y)
    # return lp + loglikelihood(theta, y, x, err_y, err_x)
    return lp + logchi(theta, y, x, err_y)


Nens = 300 #300 # number of ensemble points
ndims = 2
Nburnin = 100  #500 # number of burn-in samples
Nsamples = 3000  #500 # number of final posterior samples

# velocities=['Vout','Vout_normalized','Vout_st','Vout_st_normalized']
velocities=['Vout_Re','Vout_st_Re',]
for v in velocities:
    y, err_y, x, err_x= np.loadtxt('newest/'+v+".txt", unpack=True)
    n=len(y)
    # argslist = (y, x, err_y, err_x)
    argslist = (y, x, err_y, err_x)

    p0 = []
    for i in range(Nens):
        pi = [
            np.random.uniform(min_,max_), 
            np.random.uniform(min_,max_),
        ]
        p0.append(pi)

    # set up the sampler    
    sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)
    # pass the initial samples and total number of samples required
    t0 = time.time() # start time
    sampler.run_mcmc(p0, Nsamples + Nburnin,progress=True);
    t1 = time.time()
    timeemcee = (t1-t0)
    print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

    # extract the samples (removing the burn-in)
    samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)

    #plots
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    m_final = np.percentile(flat_samples[:, 0], [16, 50, 84])[1]
    c_final = np.percentile(flat_samples[:, 1], [16, 50, 84])[1]
    Med_value = [m_final,c_final]

    figure = corner.corner(
        flat_samples,
        # title_fmt=".2E",
        levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), 
        # levels=(0.68,0.95,0.99), 
        labels=[r"Slope", r"Intercept"], 
        quantiles=[0.16,0.84], 
        
        # range=None if 'st' in v else [(-0.6,0.6), (9.4,11.3)],
        show_titles=True, 
        label_kwargs={"fontsize": 12},
        title_kwargs={"fontsize": 10}
    );

    axes = np.array(figure.axes).reshape((ndims, ndims))
    for i in range(ndims):
        ax = axes[i, i]
        ax.axvline(Med_value[i], color="r")
    for yi in range(ndims):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(Med_value[xi], color="r")
            ax.axhline(Med_value[yi], color="r")
            ax.plot(Med_value[xi], Med_value[yi], "sr")
    # figure.savefig('figs_updatedmass/chi_corner_'+v+'.png',format='png', dpi=300)
    # figure = corner.corner(samples, levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), labels=[r"Slope", r"Intercept", r"Intrinsic Scatter", r"Intrinsic Scatter"], quantiles=[0.16,0.84], show_titles=True, label_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 10}, range=[(a_ML-0.4,a_ML+0.4), (b_ML-0.6,b_ML+0.6), (s_ML-0.1,s_ML+0.1)])

    line,hi,lo=[],[],[]
    results = '\n\n'+v+' at time '+(time.asctime( time.localtime(time.time()) )[11:19])+'\n'+'\n' + v
    labels=['slope = ','intercept = ','intrinsic scatter = ',]
    for i in range(ndims):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        # print(round(mcmc[1],3), round(q[0],3), round(-q[1],3))
        print(labels[i],round(mcmc[1],3), round(q[0],4), round(-q[1],4))
        results+='&$'+str(round(mcmc[1],3))+ '^{+'+str(round(q[0],4))+'}_{'+ str(round(-q[1],4))+'}$&'
        line.append(mcmc[1])
        hi.append(q[0])
        lo.append(q[1])
        # display(Math(txt))
    
    # fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    # samples = sampler.get_chain()
    # labels = ["m", "c",]
    # for i in range(ndims):
    #     ax = axes[i]
    #     ax.plot(samples[:, :, i], "k", alpha=0.3)
    #     ax.set_xlim(0, len(samples))
    #     ax.set_ylabel(labels[i])
    #     ax.yaxis.set_label_coords(-0.1, 0.5)
    # axes[-1].set_xlabel("step number");
    N=int(len(y)/10)-1
    chi = np.sum((y[::N]-m_final*x[::N]-c_final)**2/(err_y[::N])**2)
    print(chi,len(y[::N]))
    results+=str(chi)
    with open('results.txt','a') as f:
        f.write(results)
    plt.figure()
    x_line=np.linspace(min(x)-0.5,max(x)+0.5)
    plt.plot(x_line, line[0]*x_line+line[1],'-',color='darkorange',linewidth=4)
    plt.plot(x_line, (line[0]+hi[0])*x_line+line[1]+hi[1],'--',color='darkorange')
    plt.plot(x_line, (line[0]-lo[0])*x_line+line[1]-lo[1],'--',color='darkorange')
    plt.fill_between (x_line, (line[0]+hi[0])*x_line+line[1]+hi[1], (line[0]-lo[0])*x_line+line[1]-lo[1], color='peachpuff', hatch='\\\\\\\\', alpha=0.5, zorder=0, )
    plt.ylim(8.0, 12.0)
    plt.xlim(min(x)-0.5,max(x)+0.5)
    X,Y= x,y
    Xerr,Yerr =err_x, err_y
    Y1,Y1_e,X1,X1_e = np.loadtxt("newest/"+v+'.txt', unpack=True)
    plt.errorbar(X1, Y1,Y1_e,X1_e, fmt='h', ms=5, color='orangered', mfc='orange', mew=1, ecolor='orange', alpha=0.5, capsize=2.0, zorder=0, label='Individual Data');
    # plt.errorbar(X, Y, xerr=Xerr, yerr=Yerr, fmt='h', ms=10, color='orangered', mfc='orange', mew=1, ecolor='gray', alpha=1, capsize=2.0, zorder=5, label='Binned Data');
    # plt.errorbar(X, Y, fmt='h', ms=12, color='orangered', mfc='none', mew=1, ecolor='gray', alpha=1, capsize=2.0, zorder=6);

    plt.tick_params(direction='inout', length=7, width=2)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('$log V_c$')
    if 'st' in v: ylabel = '$log M_{star}$'
    else: ylabel = '$log M_{bar}$'
    plt.title('m={}, b={}'.format(round(line[0],2),round(line[1],2)))
    plt.ylabel(ylabel)
    # plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.,fontsize=11 )
    plt.legend()
    # plt.savefig('figs_updatedmass/chi_bestfit_'+v+'.png')
    # plt.show()
