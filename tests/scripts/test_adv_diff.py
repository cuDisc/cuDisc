import os 
from fileIO import *
import numpy as np
import matplotlib.pyplot as plt

print("Making and running test...\n")

os.system('make -j -C ../../ test_adv_diff')
os.system('../../test_adv_diff')

sim1 = CuDiscModel("../outputs/adv_diff/run_128/")
sim2 = CuDiscModel("../outputs/adv_diff/run_256/")
sim3 = CuDiscModel("../outputs/adv_diff/run_512/")
# sim4 = CuDiscModel("./run_1024/")

gas1, dust1 = sim1.load_all_prim_data()
gas2, dust2 = sim2.load_all_prim_data()
gas3, dust3 = sim3.load_all_prim_data()
# gas4, dust4 = sim4.load_all_prim_data()

print("Generating plots...")

fig, ax = plt.subplots(1,3,gridspec_kw={'width_ratios': [1, 1, .05]}, figsize=(6,2.5))
for i in range(2):
    idx = 100 + dust2.rho.shape[0]*10 + i+1
    # ax = plt.subplot(idx)
    ax[i].set_aspect('equal')
    cont = ax[i].contourf(sim2.grid.R, sim2.grid.Z, np.log10(dust2.rho[i,:,:,0]),np.linspace(-4,1,100), cmap = 'YlOrRd', extend='both')
    ax[i].set_xlim(27.5, 41.5)
    ax[i].set_ylim(-5, 9)
    ax[i].set_xlabel("X (Arb. units)")
    if i == 0:
        ax[i].set_ylabel("Y (Arb. units)")


cbar = plt.colorbar(mappable=cont,cax=ax[2])
cbar.set_label("Density (Arb. units)")
cbar.set_ticks([-4,-3,-2,-1,0,1])
cbar.set_ticklabels([10**(-4),10**(-3),10**(-2),10**(-1),10**(0),10**(1)])
plt.tight_layout()
plt.savefig("../outputs/adv_diff/2dgau.png")

def P(x,y,t,D,A):
    return A/(t) * np.exp(-(x-30-(5*(t-0.1)))**2/(4*D*t)) * np.exp(-(y-(2*(t-0.1)))**2/(4*D*t))

fig = plt.figure(figsize=(6,4))
sims = [sim1,sim2,sim3]#,sim4]
dusts = [dust1,dust2,dust3]#,dust4]
ns = np.array([128,256,512])#,1024])
L2s = np.zeros(len(sims))
for i in range(len(sims)):
    RMS_error = (dusts[i].rho[-1,2:-2,2:-2,0]-P(sims[i].grid.R[2:-2,2:-2],sims[i].grid.Z[2:-2,2:-2],1.,1.,1.))**2./((sims[i].grid.NR-4)*(sims[i].grid.NZ-4))
    L2s[i] = np.sqrt(np.sum(RMS_error))

plt.loglog(ns, L2s, linestyle='--',marker='x', label='L2 error')
plt.loglog(ns, 50/ns**2, label='1/N$^2$')
plt.loglog(ns, 0.5/ns, label='1/N')
plt.xlabel("Number of cells, N")
plt.ylabel("L2 error norm")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/adv_diff/advdiff_error.png")

print("Done!\n")
