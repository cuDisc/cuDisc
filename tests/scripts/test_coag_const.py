from fileIO import *
import numpy as np
import matplotlib.pyplot as plt
from constants import *

if os.path.exists("../outputs/coag_const/coag_const_4.txt"):
    print("Test completed.")
else:
    print("Making and running test...\n")

    os.system('make -j -C ../../ test_coagconst')
    os.system('../../test_coagconst')

class ConstSolution(object):
    '''Assympotic solution for constant coagulation kernel,
           K(i,j) = K0
           
    The solution is exact for an initial exponential distribution. The slope of
    the exponential distribution is unimportant at large masses and large times.

    args: 
        K0  : float, kernel coefficient
        m0  : float, total mass
        mu  : float, slope of the initial mass distribution 
    '''
    def __init__(self, K0=4., m0=1., mu=1.):
        self._K0 = K0
        self._m0 = m0
        self._mu = mu
        self._N0 = mu * m0 / 2.

        self._a = 0.5 * K0 * self._N0

    def _f(self, t):
        return 1 - 1 / (1 + self._a * t)

    def _g(self, t):
        return 1 / (1 + self._a * t)**2

    def _sol(self, m, t):
        '''Solution for N(m, t)'''
        f2 = self._f(t)**0.5
        g = self._g(t)

        C = 0.5 * (self._N0 * g / f2) * self._mu 

        return C * (np.exp(-self._mu * (1 - f2) * m) - 
                    np.exp(-self._mu * (1 + f2) * m))

    def _sol2(self, m, t):
        '''Solution for a*t << 1'''
        g = self._g(t)
        mmu = self._mu * m
        C = self._N0 * g * mmu  
        return C * np.exp(-mmu) * (1 + mmu * mmu * self._f(t) / 6) 



    def assymptote(self, m, t):
        '''Solution for large m and t'''
        at = self._a * t
        C = self._mu / (2 * at)

        C = (self._N0/at) * C * np.exp(-m*C)
        return C * ( 1 - np.exp(-2*self._mu *m))

    def __call__(self, m, t):
        '''Solution for N(m, t)'''
        if self._a*t > 1e-3:
            return self._sol(m, t)
        else:
            return self._sol2(m,t)

    @property
    def tg(self):
        return 1. / self._a

sim = CuDiscModel("../outputs/coag_const")

sizes = sim.load_grain_sizes()
g = sim.grid
rhos = np.zeros((5,g.R.shape[0],g.R.shape[1],sizes.a_c.shape[0]))
for i in range(5):
    rhos[i] = np.loadtxt("../outputs/coag_const/coag_const_"+str(i)+".txt").reshape(g.NR,g.NZ,sizes.a_c.shape[0])

mtot = rhos[0,0,0].sum()
Sol = ConstSolution(m0=mtot)

print("Generating plot...")

t = [0, 1, 10, 100, 1000]
dNdm = rhos[:,0,0] / (np.diff(sizes.m_e)*sizes.m_c)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

plt.figure(figsize=(6,4))
for i in range(5):
    plt.loglog(sizes.a_c, dNdm[i], c=colors[i], label='t = %d'%t[i])
    if i==4:
        plt.loglog(sizes.a_c, Sol(sizes.m_c, t[i]), ls=(0,(5,5)), c='black', label='Analytic')
    else:
        plt.loglog(sizes.a_c, Sol(sizes.m_c, t[i]), ls=(0,(5,5)), c='black')

plt.legend()
plt.ylim(1e-12,1)
plt.xlim(1e-2,2e2)
plt.xlabel("Grain size (arb.)")
plt.ylabel("Density (arb.)")
plt.tight_layout()
plt.savefig("../outputs/coag_const/coag_const.png")

print("Done!\n")

