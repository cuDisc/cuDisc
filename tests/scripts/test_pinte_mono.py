import os
from constants import *
from fileIO import *
import numpy as np
import matplotlib.pyplot as plt

print("Making and running test...\n")

os.system('cd ../../ && make -j test_pinte_graindist_mono')
os.system('../../test_pinte_graindist_mono')

sim1 = CuDiscModel("../outputs/pinte_mono/run_thick")
sim2 = CuDiscModel("../outputs/pinte_mono/run_thin")

print("Generating plot...")

g_p = sim1.grid
T = sim1.load_temp("0").T

r_rad, th_rad = np.meshgrid(g_p.R[:,0], np.pi/2. - np.flip(np.arctan(g_p.tan_th_c[1:])), indexing='ij')

g_p2 = sim2.grid
T2 = sim2.load_temp("0").T

r_rad2, th_rad2 = np.meshgrid(g_p2.R[:,0], np.pi/2. - np.flip(np.arctan(g_p2.tan_th_c[1:])), indexing='ij')

""" RADMC3D data needed for comparison (too large for GitHub, ask authors if desired)"""

# temp = np.loadtxt("./radmc_data_for_pinte/radmc_thin.dat", skiprows=3)
# temp1 = np.loadtxt("./radmc_data_for_pinte/radmc_thick.dat", skiprows=3)
# temp = temp.reshape([100,r_rad2.shape[1],r_rad2.shape[0]]).T
# temp1 = temp1.reshape([100,r_rad.shape[1],r_rad.shape[0]]).T

fig,ax = plt.subplots(1,2,figsize = (14,4), sharey=True )
fig.subplots_adjust(wspace=0.05, hspace=0)
# ax[0].loglog((r_rad2[:,0]*np.cos(np.arctan(g_p2.tan_th_c[0+1]))-r_rad2[0,0]*np.cos(np.arctan(g_p2.tan_th_c[0+1])))/au, np.flip(temp[:,:,0],1)[:,0], c='black', ls='--', label = 'RADMC')
ax[0].loglog((g_p2.R[:,0]-g_p2.R[0,0])/au, T2[:,0+1], c='r', label='cuDisc, mid-plane')
# ax[0].loglog((r_rad2[:,0]*np.cos(np.arctan(g_p2.tan_th_c[120+1]))-r_rad2[0,0]*np.cos(np.arctan(g_p2.tan_th_c[120+1])))/au, np.flip(temp[:,:,0],1)[:,120],c='black', ls='--',)
ax[0].loglog((g_p2.R[:,0]-g_p2.R[0,0])/au, T2[:,120+1], c='g', label='cuDisc, surface')
# ax[1].loglog((r_rad[:,0]*np.cos(np.arctan(g_p.tan_th_c[0+1]))-r_rad[0,0]*np.cos(np.arctan(g_p.tan_th_c[0+1])))/au, np.flip(temp1[:,:,0],1)[:,0], c='black', ls='--', label = 'RADMC')
ax[1].loglog((g_p.R[:,0]-g_p.R[0,0])/au, T[:,0+1], c='r', label='FLD')
# ax[1].loglog((r_rad[:,0]*np.cos(np.arctan(g_p.tan_th_c[120+1]))-r_rad[0,0]*np.cos(np.arctan(g_p.tan_th_c[120+1])))/au, np.flip(temp1[:,:,0],1)[:,120],c='black', ls='--',)
ax[1].loglog((g_p.R[:,0]-g_p.R[0,0])/au, T[:,120+1], c='g')
ax[0].set_xlim(1e-6,400)
ax[1].set_xlim(1e-6,400)
ax[0].set_ylabel('T (K)')
ax[0].set_xlabel('R - R$_{in}$ (AU)')
ax[1].set_xlabel('R - R$_{in}$ (AU)')
ax[0].legend()

ax[0].text(1,750,r"$1\times 10^{-2}$ $M_\oplus$", fontsize = 16)
ax[1].text(5,750,r"10 $M_\oplus$", fontsize = 16)

plt.savefig("../outputs/pinte_mono/tempcomp.png",bbox_inches='tight')
print("Done!\n")