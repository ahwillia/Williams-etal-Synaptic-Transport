# Download NEURON: http://www.neuron.yale.edu/neuron/download
# Download PyNeuronToolbox: https://github.com/ahwillia/PyNeuron-Toolbox

from __future__ import division
from neuron import h
import numpy as np
from scipy.linalg import expm  # matrix exponential used to solve linear system
import pylab as plt
from matplotlib import animation
from matplotlib.pyplot import cm
np.random.seed(123456789)

## Get a list of segments
from PyNeuronToolbox.morphology import shapeplot,allsec_preorder,root_indices,shapeplot_animate

def get_nsegs(h):
    N = 0
    for sec in h.allsec():
        N += sec.nseg
    return int(N)

def sushi_system(h,a,b,c):
    """
    Returns a matrix A, such that dx/dt = A*x
    
    N = # of compartments
    A is (2N x 2N) matrix
    x is (2N x 1) vector.
      The first N elements correspond to concentrations of u (molecules in transit)
      The second half correspond to concentrations of u-star (detached/active molecules)
    The trafficking rate constants along the microtubules are given by the vectors "a" and "b"
    The rate constants for u detaching/attaching (turning into u*) are given by "c" and "d"
    """

    N = len(c)
    sec_list = allsec_preorder(h)
    
    ## State-space equations
    #  dx/dt = Ax + Bu
    A = np.zeros((2*N,2*N))

    # Trafficking along belt
    # Iterative traversal of dendritic tree in pre-order
    i = 0
    section = None
    parentStack = [(None,sec_list[0])]
    while len(parentStack)>0:
        # Get next section to traverse
        #  --> p is parent index, section is h.Section object
        (p,section) = parentStack.pop()
        
        # Trafficking to/from parent
        if p is not None:
            # Out of parent, into child
            ai = a.pop()
            A[p,p] += -ai
            A[i,p] += ai
            # Into parent, out of child
            bi = b.pop()
            A[p,i] += bi
            A[i,i] += -bi
        
        # visit all segments in compartment
        for (j,seg) in enumerate(section):
            # Deal with out/into rates within compartment, just tridiag matrix
            if j>0:
                # Out of parent, into child
                ai = a.pop()
                A[i-1,i-1] += -ai
                A[i,i-1] += ai
                # Into parent, out of child
                bi = b.pop()
                A[i-1,i] += bi
                A[i,i] += -bi
            # move onto next compartment
            i += 1
        
        # now visit children in pre-order
        child_list = list(h.SectionRef(sec=section).child)
        if child_list is not None:
            child_list.reverse()
        for c_sec in child_list:
            parentStack.append([i-1,c_sec]) # append parent index and child
    
    Abelt = np.copy(A[:N,:N])

    # Detachment off the belt
    for i in range(N):
        A[i,i] += -c[i]
        A[i+N,i] += c[i]
    
    return Abelt, A

def set_trafficking_rates(h, utarg, diff_coeff):
    """
    (a+b) = 2 * diff_coeff / dist_between(p,i)
    """
    N = len(utarg)
    a,b = [],[]
    sec_list = allsec_preorder(h)
    
    # Iterative traversal of dendritic tree in pre-order
    i = 0
    parentStack = [(None,None,sec_list[0])]
    while len(parentStack)>0:
        # Get next section to traverse
        #  --> p is parent index, section is h.Section object
        (p,psize,section) = parentStack.pop()
        secsize = section.L / section.nseg
        
        # Trafficking to/from parent
        if p is not None:
            mp = utarg[p] # concentration in parent
            mc = utarg[i] # concentration in child
            limit = 2.0 * diff_coeff / ((0.5*(psize+secsize))**2)
            a.insert(0, limit / (1.0 + mp/mc) )
            b.insert(0, limit / (1.0 + mc/mp) )
        
        # visit all segments in section
        for (j,seg) in enumerate(section):
            # Deal with out/into rates within compartment, just tridiag matrix
            if j>0:
                mp = utarg[i-1]
                mc = utarg[i]
                limit = 2.0 * diff_coeff / (secsize**2)
                a.insert(0, limit / (1.0 + mp/mc) )
                b.insert(0, limit / (1.0 + mc/mp) )
                
            # move onto next compartment
            i += 1
        
        # now visit children in pre-order
        child_list = list(h.SectionRef(sec=section).child)
        if child_list is not None:
            child_list.reverse() # needed to visit children in correct order
        for c_sec in child_list:
            parentStack.append((i-1,secsize,c_sec)) # append parent index and child

    return a,b

def run_uniform_sim(h, cscale, diff_coeff, belt_only=False, **kwargs):
    N = get_nsegs(h)
    a,b = set_trafficking_rates(h, np.ones(N), diff_coeff)
    c = list(np.ones(N)*cscale)
    Abelt,A = sushi_system(h,a,b,c)
    if belt_only:
        u,t = run_sim(h,Abelt,**kwargs)
        return Abelt,u,t
    else:
        u,t = run_sim(h,A,**kwargs)
        targ = np.sum(u[0,:])/N
        err = 100*np.mean( np.abs(u[:,N:] - targ ) / targ ,axis=1)
        return A,u,t,list(err)

def run_sim(h, A, t0=2e1, tmax=5e7, dt=2):
    u0 = np.zeros(A.shape[0])
    N = float(A.shape[0]/2)
    roots = root_indices(allsec_preorder(h))
    for r in roots: 
        u0[r] = N / len(roots)
    u = [u0,np.dot(expm(t0*A),u0)]
    t = [0,t0]
    while t[-1] < tmax:
        t.append(t[-1]*dt)
        u.append(np.dot(expm(t[-1]*A),u0))
    
    return np.array(u),np.array(t)

def plot_steady_state(A,filename,view,tol=0.1):
    tss = 1e6
    N = int(A.shape[0]/2)
    u0 = np.zeros(N*2)
    u0[0] = N
    u = np.dot(expm(tss*A),u0)
    while np.sum(u[N:])<(N*(1-tol)):
        tss *= 10
        u = np.dot(expm(tss*A),u0)

    np.savetxt('./data/'+filename, u[N:])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    clim = [0,2]
    shapeplot(h, ax, cvals=u[N:], cmap=plt.cm.cool, clim=clim)
    ax.view_init(*view)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(clim[0],clim[1])) 
    sm._A = []
    plt.colorbar(sm, shrink=0.5)
    plt.tight_layout()
    plt.savefig('./plots/'+filename+'.png')

def save_movie(h, t, u, view, filename, clim=[0,2]):
    # Make an animation
    fig = plt.figure(figsize=(8,8))
    shapeax = plt.subplot(111, projection='3d')
    lines = shapeplot(h,shapeax,order='pre',lw=2)
    shapeax.view_init(*view)
    plt.title('cargo distribution over (log) time',fontweight='bold',fontsize=14)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(clim[0],clim[1])) 
    sm._A = []
    plt.colorbar(sm, shrink=0.5)
    shapeax.set_axis_off()
    plt.tight_layout()

    anim = None

    #anim_func = shapeplot_animate(u,lines,clim=clim,cmap=cm.cool)
    #anim = animation.FuncAnimation(fig, anim_func, frames=u.shape[0], interval=400, blit=True)
    #anim.save('./anim/'+filename+'.mp4', fps=30)
    return anim

def run_sims(h,cellname,view):
    diff_coeff = 4.0 
    for cscale in [1e-2,1e-3,1e-4,1e-5,1e-6]:
        A,u,t,err = run_uniform_sim(h, cscale, diff_coeff)
        return A,u,t,err
        # N = A.shape[0]/2

        # filename = cellname+'_1e'+str(int(np.log10(cscale)))
        # plot_steady_state(A, filename, view)
        # anim = save_movie(h, t, u[:,:N], view, filename+'_belt')
        # return anim
        #save_movie(h, t, u[:,N:], view, filename+'_detached')