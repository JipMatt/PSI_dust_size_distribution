'''
Code: data_prep.py
Credit: Jip Matthijsse 2025
License: GNU GENERAL Public License (https://www.gnu.org/licenses/)

Code underlying the manuscript: Matthijsse, J. & Aly, H., & Paardekooper, S.-J., 2025
"Polydisperse Formation of Planetesimals: The dust size distribution in clumps"

This code generates the processed data from the raw output from the FARGO3D simulations.
'''

###############################################################################
###############################################################################
"""
Set Directory path to the raw output from FARGO3D
"""
#SET DIRECTORY: file_dir = '/file/path/to/data'
file_dir = "/file/path/to/data"
file_dir = '/Volumes/staff-umbrella/datajmatthijsse'
###############################################################################
###############################################################################

from polydust import Polydust, SizeDistribution
import vistools as vt

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

# Gauss-Legendre collocation points
from scipy.special import roots_legendre
import scipy.stats as spstat
from scipy.interpolate import BarycentricInterpolator

######################################################
######################################################
######################################################

#paper
# Function to adjust color brightness
def adjust_brightness(color, factor):
    rgb = mcolors.hex2color(color)
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv = (hsv[0],  min(1.0, hsv[1] * 1/factor), min(1.0, hsv[2] * factor))  # Ensure brightness stays within [0, 1]
    return mcolors.hsv_to_rgb(hsv)

medium =sns.color_palette("colorblind")
medium[5] = sns.color_palette("colorblind")[7]

light = [mcolors.to_hex(adjust_brightness(color, 1.33)) for color in medium]
dark = [mcolors.to_hex(adjust_brightness(color, 1/1.33)) for color in medium]
light[5] = sns.color_palette("pastel")[7]

lwidth=.75
cmap = sns.color_palette("flare", as_cmap=True)
figsize2 = (3.54331, 3.54331/(5/3))
figsize3= (7.48031, 7.48031/1.85)
colorback = 'white'#(32.2/100, 2/100, 48.2/100)
colortxt = 'black'#(88.2/100, 91/100, 92.2/100)
plt.rc('figure', figsize=(3.54331, 3.54331/(16/9)), dpi=300, facecolor=colorback, edgecolor=colorback)
plt.rc('savefig', bbox='tight')
plt.rc('lines', linewidth=lwidth)
plt.rc('xtick', labelsize=7, color=colortxt, top=True) 
plt.rc('xtick.major', size=3.5*lwidth, width=0.8*lwidth)
plt.rc('ytick.major', size=3.5*lwidth, width=0.8*lwidth)
plt.rc('xtick.minor', size=2*lwidth, width=0.6*lwidth)
plt.rc('ytick.minor', size=2*lwidth, width=0.6*lwidth)
plt.rc('ytick', labelsize=7, color=colortxt, right=True)
plt.rc('font', size=7, family='serif', serif='Times New Roman')
plt.rc('mathtext', fontset= 'dejavuserif')
plt.rc('axes', facecolor=colorback, edgecolor=colortxt, titlesize=7, titlecolor=colortxt,  labelsize=7, labelcolor=colortxt, linewidth=.8*lwidth) #fontsize of the title #fontsize of the x and y labels
plt.rc('legend', labelcolor=colortxt, fontsize=5, title_fontsize=4, framealpha=0.6, markerscale = .6) #fontsize of the legend

plt.rc('axes', grid=True, ) #grid
plt.rc('axes.grid', which = 'both')
plt.rc('grid', color='#cccccc', linewidth=  0.25*lwidth)

######################################################
######################################################
######################################################

try:
    from psitools.psi_mode import PSIMode
except ImportError as error:
    # as of python 3.6 now throws ModuleNotFoundError
    print('Will run despite error importing psitools:', error)
    print('If PSIMode is needed, install psitools')

def PSI_eigen(dust_to_gas_ratio, stokes_range, wave_number_x, wave_number_z,
              viscous_alpha=0.0, size_distribution_power=3.5, mono=False):
    '''Calculate PSI eigenfunctions

    Args:
        dust_to_gas_ratio: Background dust to gas ratio
        stokes_range: minimum and maximum Stokes number
        wave_number_x: Kx
        wave_number_z: Kz
        viscous_alpha (optional): viscosity parameter, defaults to zero.
    '''
    # Use PSIMode to calculate eigenvalue
    np.random.seed(0)
    pm = PSIMode(dust_to_gas_ratio=dust_to_gas_ratio,
                 stokes_range=stokes_range,
                 real_range=[-5, 5],
                 imag_range=[1e-10, 1],
                 size_distribution_power=size_distribution_power, single_size_flag=mono)

    roots = pm.calculate(wave_number_x=wave_number_x,
                         wave_number_z=wave_number_z,
                         viscous_alpha=viscous_alpha,
                         constant_schmidt=True)
    if len(roots) ==0:
        roots = [0]

    return roots

def distribution(stokes_range = [1e-3, 1e-1], dust_density = 3, n_dust = 1e3, gas_density = 1, viscous_alpha=0.0, slope=None, k=30):
    '''
    Calculetes the intrisic parameters of the initial conditions
    '''
    size_dist = SizeDistribution(stokes_range)

    # Create polydust object
    pdt = Polydust(n_dust=n_dust,
                stokes_range=stokes_range,
                dust_density=dust_density,
                gas_density=gas_density,
                size_distribution=size_dist,
                gauss_legendre=True,
                discrete_equilibrium=False)
    if slope== 3.2:
        pdt.sigma = pdt.size_distribution.sigma32
    if slope== 3.8:
        pdt.sigma = pdt.size_distribution.sigma38

    tau, sigmas, vel = pdt.initial_conditions()
    velx = vel[np.arange(len(vel))%3==0]
    avg_stokes_0 = np.sum(tau*sigmas)/np.sum(sigmas)

    #calculate resonant of radial drift from distribution
    mu = dust_density/gas_density
    sigma = dust_density
    AN = lambda tau: mu*sigma*tau/(1 + tau*tau)
    BN = lambda tau: 1.0 + mu*sigma/(1 + tau*tau)
    func = lambda tau: 2*AN(tau)/(AN(tau)*AN(tau) + BN(tau)*BN(tau)) - velx[0]
    ts_mono_vres= sp.optimize.fsolve(func, avg_stokes_0)[0]


    if slope is None:
        slope =3.5
    omega = PSI_eigen(dust_to_gas_ratio=mu,
                                    stokes_range=stokes_range,
                                    wave_number_x=k,
                                    wave_number_z=k, viscous_alpha=viscous_alpha, size_distribution_power=slope, mono=False)
    
    omega_mono = PSI_eigen(dust_to_gas_ratio=mu,
                                    stokes_range=stokes_range,
                                    wave_number_x=k,
                                    wave_number_z=k, viscous_alpha=viscous_alpha, size_distribution_power=slope, mono=True)
    if len(omega) != 1:
        omega = [(0+0j)]
    if len(omega_mono) != 1:
        omega_mono = [(0+0j)]
    return ts_mono_vres, avg_stokes_0, omega[0].imag, omega_mono[0].imag

ts_e3_e1_vres, ts_e3_e1_avg0, ts_e3_e1_growth, ts_e1_growth = distribution()
ts_e3_e1_vres, ts_e3_e1_avg0, ts_e3_e1_growth_alpha8, ts_e1_growth_alpha8 = distribution(viscous_alpha=1e-8)
ts_e3_e1_vres, ts_e3_e1_avg0, ts_e3_e1_growth_alpha7, ts_e1_growth_alpha7 = distribution(viscous_alpha=1e-7)
ts_e3_e1_vres, ts_e3_e1_avg0, ts_e3_e1_growth_alpha6, ts_e1_growth_alpha6 = distribution(viscous_alpha=1e-6)
ts_e4_e1_vres, ts_e4_e1_avg0, ts_e4_e1_growth, ts_e1_growth2  = distribution(stokes_range=[1e-4,1e-1])
ts_e3_5e2_vres, ts_e3_5e2_avg0, ts_e3_5e2_growth, ts_5e2_growth = distribution(stokes_range=[1e-3,5e-2])
ts_e3_2e1_vres, ts_e3_2e1_avg0, ts_e3_2e1_growth, ts_2e1_growth = distribution(stokes_range=[1e-3,2e-1])

ts_e3_e1_vres_mrn32, ts_e3_e1_avg0_mrn32, ts_e3_e1_growth_mrn32, ts_e1_growth2 = distribution(slope=3.2)
ts_e3_e1_vres_mrn38, ts_e3_e1_avg0_mrn38, ts_e3_e1_growth_mrn38, ts_e1_growth2 = distribution(slope=3.8)

ts_e3_e1_vres_mu1, ts_e3_e1_avg0_mu1, ts_e3_e1_growth_mu1, ts_e1_growth_mu1 = distribution(dust_density=1)
ts_e3_e1_vres_mu05, ts_e3_e1_avg0_mu05, ts_e3_e1_growth_mu05, ts_e1_growth_mu05 = distribution(dust_density=.5)
ts_e3_e1_vres_k10, ts_e3_e1_avg0_k10, ts_e3_e1_growth_k10, ts_e1_growth_k10 = distribution(k=10)

###########################################
d_dust = 3
x = np.logspace(-3, -1, 1000)
sigma_ts0= lambda tl, tu, x: ((d_dust/2)/(tu**0.5 - tl**.5))*x**(-0.5)
sigma_ts0_32= lambda tl, tu, x: ((d_dust/5)/(tu**(1/5) - tl**(1/5)))*x**(-4/5)
sigma_ts0_38= lambda tl, tu, x: ((d_dust*4/5)/(tu**(4/5) - tl**(4/5)))*x**(-1/5)

######################################################
######################################################
######################################################

data = pd.read_csv(file_dir+"/processed_data/fitted.csv", index_col=0).to_dict('list')
growth = pd.read_csv(file_dir+"/processed_data/growthrates_ts.csv", index_col=0)

fig, axs = plt.subplots(2,1, sharex=True, figsize=(3.54331, 3.54331))
fig.subplots_adjust(hspace=0)

ax1 =axs[0]
ax1.vlines(0.1, 1e-5, 1e0, color='k', alpha=.5)

ax1.vlines(0.037, 1e-5, 1e0, color='k', linestyles='dotted', alpha=.5)

ax1.plot(growth['tsmax'], growth['growth_mono'], '-o', markersize=.5, color='tab:orange')
ax1.plot(growth.tsmax[growth.growth_poly != 0], growth.growth_poly[growth.growth_poly != 0], '-o', markersize=.5, color='tab:blue')
ax1.hlines(ts_e3_e1_growth, 0.005, 0.5,  colors='k', alpha=.5)

ax1.plot([0,0], '-o', markersize=.5, color='darkgrey', label='Analytical')
ax1.errorbar(data['tsmaxs'], data['l_mono_slope'], yerr=data['l_mono_slope_std'], fmt='D', markersize=2, color='orangered', capsize=2.5, label='mSI')
ax1.errorbar(data['tsmaxs'], data['l_poly_slope'], yerr=data['l_poly_slope_std'], fmt='D', markersize=2, color='darkblue',capsize=2.5, label='PSI')
ax1.set_xlabel(r'$\tau_\mathrm{s, max}$')
ax1.set_ylabel(r'$\Im(\omega)$')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(1e-5, 1e0)
ax1.set_xlim(1e-2, 5e-1)
ax1.legend(loc='lower right')

# plt.xlim(5e-3,1e-1)
ax2 = axs[1]
ax2.vlines(0.1, 1e0, 9e1, color='k', alpha=.5, label=r'PSI $\tau_\mathrm{s, max}=0.1$')
ax2.vlines(0.037, 1e0, 9e1, color='k', linestyles='dotted', alpha=.5, label=r'Avg. $\bar{\tau}_\mathrm{s}=0.037$')

ax2.set_ylabel(r'$\rho_\mathrm{d \, max}/\bar{\rho}^0_\mathrm{d}$')
ax2.hlines(data['l_poly_max'][2], 0.005, 0.5, colors='k', alpha=.5)
ax2.fill_between([0.005, 0.5], [data['l_poly_max'][2]+data['l_poly_max_std'][2], data['l_poly_max'][2]+data['l_poly_max_std'][2]], [data['l_poly_max'][2]-data['l_poly_max_std'][2], data['l_poly_max'][2]-data['l_poly_max_std'][2]], color='grey', alpha=0.33)
ax2.errorbar(data['tsmaxs'], data['l_poly_max'], yerr=data['l_poly_max_std'], fmt='D', markersize=2,color='darkblue', capsize=2.5)
ax2.errorbar(data['tsmaxs'], data['l_mono_max'], yerr=data['l_mono_max_std'], fmt='D', markersize=2,color='orangered', capsize=2.5)

ax2.set_yscale('log')
ax2.set_ylim(1e0, 9e1)
ax2.set_xlabel(r'$\tau_\mathrm{s, max}$')
ax2.legend(loc='lower right')


legend_elments = [Patch(visible=False, label=r'PSI'),
                  Line2D([0], [0], color=colortxt,  label=r'Dust'),
                  Line2D([0], [0], color=colortxt, linestyle='dotted',  label=r'Sum-Dust'),
                  Patch(visible=False, label=r'mSI'),
                  Line2D([0], [0], color=colortxt, linestyle='dashdot',  label=r'mSI')]

fig.savefig('tau_maxs.png')
plt.close()

######################################################
######################################################
######################################################

dfpoly = pd.read_csv(file_dir+"/processed_data/res1024_poly10_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly.groupby(level=['ts']).transform('first'))
dfpoly0 = dfpoly / t0

fluid = 'sum-dust'
filtered = 'fluid == @fluid'
dfmono = pd.read_csv(file_dir+"/processed_data/res1024_mono_ts(1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
dfmono = dfmono.query(filtered)
t0 = (dfmono.groupby(level=['ts']).transform('first'))
dfmono0 = dfmono / t0

######################################################

ts = dfpoly.index.get_level_values('ts')
norm = mcolors.LogNorm(1e-3, 1e-1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
fig =  plt.figure(figsize=(3.411,3.411/(.5)))
gs_main = gridspec.GridSpec(5,1, figure=fig, height_ratios=[1, 1,1,.3,1], hspace=0)#[

ax1 = fig.add_subplot(gs_main[0])

g = sns.lineplot(data=dfpoly, x="t", y="ampfft", hue='ts', style='fluid', palette=cmap, hue_norm=norm, legend=False, ax=ax1)
sns.lineplot(data=dfmono, x="t", y="ampfft", color=sm.to_rgba(1e-1),linestyle='dashdot',ax=ax1)
t = np.linspace(0, 150, 100)
ax1.plot(t, 1.5e-5*np.exp(ts_e3_e1_growth*t), color=colortxt, linewidth=lwidth*.8,linestyle='dashed')

ax1.plot(t[t<30], 1.2e-6*np.exp(ts_e1_growth*t[t<30]), color=colortxt, linewidth=lwidth*.8, linestyle='dashed')

ax1.text(0.035, 0.93, 'A',
     horizontalalignment='center',
     verticalalignment='center', fontsize=6,
     transform = ax1.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

# ax.set_title('poly-10, res:1024')
ax1.set_ylabel(r'$P_K$')
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_yscale('log')
ax1.set_xlim(0, 160)
ax1.set_ylim(1e-6,1e0)

legend2 = ax1.legend(handles = [Patch(visible=False, label=r'$\mathbf{Analytical}$'),
                                Line2D([0], [0], color=colortxt, linestyle='dashed', lw=1, label=r'$\propto \exp(\Im{(\omega)\cdot t})$')], loc='lower left')
plt.gca().add_artist(legend2)

legend_elments = [Patch(visible=False, label=r'PSI'),
                  Line2D([0], [0], color=colortxt,  label=r'Dust Species'),
                  Line2D([0], [0], color=colortxt, linestyle='dotted',  label=r'Sum Dust Species'),
                  Patch(visible=False, label=''),
                  Line2D([0], [0], color=colortxt, linestyle='dashdot',  label=r'mSI')]
                #   Patch(visible=False, label=r'$\mathbf{Analytical}$'),
                #   Line2D([0], [0], color=colortxt, linestyle='dashed',  label=r'$\propto \exp[\Im(\omega) \cdot t]$')]
ax1.legend(handles=legend_elments[:], loc='lower right')

ax2 = fig.add_subplot(gs_main[1])

g = sns.lineplot(data=dfpoly0, x="t", y="max", hue='ts', style='fluid', palette=cmap, hue_norm=norm, legend=False, ax=ax2)
sns.lineplot(data=dfmono0, x="t", y="max", color=sm.to_rgba(1e-1),linestyle='dashdot',ax=ax2)
ax2.set_ylabel(r'$\rho_{\mathrm{d} \, {\mathrm{max}}} \, / \, \bar{\rho}_\mathrm{d}^0$')
plt.setp(ax2.get_xticklabels(), visible=False)

ax2.text(0.035, 0.93, 'B',
     horizontalalignment='center',
     verticalalignment='center', fontsize=6,
     transform = ax2.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

ax2.set_yscale('log')
ax2.set_xlim(0, 160)


ax3 = fig.add_subplot(gs_main[2])

g = sns.lineplot(data=dfpoly0, x="t", y="avg99", hue='ts', style='fluid', palette=cmap, hue_norm=norm, legend=False, ax=ax3)
sns.lineplot(data=dfmono0, x="t", y="avg99", color=sm.to_rgba(1e-1),linestyle='dashdot',ax=ax3)
ax3.set_ylabel(r'$\bar{\rho}_{\mathrm{d} \, {(> \!99\%)}} \, / \, \bar{\rho}^0$')
ax3.set_xlabel(r'$\Omega t$')

ax3.set_xlabel(r'$\Omega t$')
ax3.set_ylabel(r'$\bar{\rho}_{\mathrm{d} \, {(> \!99\%)}} \, / \, \bar{\rho}_\mathrm{d}^0$')

# ax.set_title('poly-10, res:1024')
ax3.set_yscale('log')
ax3.set_xlim(0, 160)
ax3.text(0.035, 0.93, 'C',
     horizontalalignment='center',
     verticalalignment='center', fontsize=6,
     transform = ax3.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))



ax4 = fig.add_subplot(gs_main[4])
filtered = '140 <= t <= 160' 
dfpoly = pd.read_csv(file_dir+"/processed_data/res1024_poly10_ts(1e-3|1e-1)_k30_hist.csv", index_col=['t','fluid','ts', 'lbin', 'cbin', 'ubin'])
dfpoly_t = dfpoly.query(filtered)


dfmono = pd.read_csv(file_dir+"/processed_data/res1024_mono_ts(1e-1)_k30_hist.csv", index_col=['t','fluid','ts', 'lbin', 'cbin', 'ubin'])
fluids = "sum-dust"
filtered = '140 <= t <= 160 & fluid == @fluids' 
dfmono_t = dfmono.query(filtered)
dfpoly.index.get_level_values('t').max()
ts = dfpoly_t.index.get_level_values('ts')
norm = mcolors.LogNorm(1e-3, 1e-1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

sns.lineplot(data=dfmono_t, x='lbin', y='pdf0',  color=colortxt, linestyle='dashed', drawstyle='steps-pre', palette=cmap, hue_norm=norm, legend=False, ax=ax4)
sns.lineplot(data=dfpoly_t, x='lbin', y='pdf0',  hue='ts', style='fluid', drawstyle='steps-pre', palette=cmap, hue_norm=norm, legend=False, ax=ax4)

ax4.text(0.035, 0.93, 'D',
     horizontalalignment='center',     verticalalignment='center', fontsize=6,
     transform = ax4.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

ax4.text(.5, 0.93, r'$140 \leq \Omega t \leq 160$',
     horizontalalignment='center',
     verticalalignment='center', fontsize=5,
     transform = ax4.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

ax4.set_xlabel(r'$\rho_\mathrm{d}/\bar{\rho}_\mathrm{d}^0$')
ax4.set_ylabel(r'$PDF$')
# ax.set_title(r'poly-10, res:512, $150 < \Omega t < 300$')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(1e-3, 1e2)
ax4.set_ylim(1e-5, 1e1)



cbar = fig.colorbar(sm, ax=[ax1 ,ax2, ax3, ax4], orientation='vertical', aspect=40)
cbar.set_label(r'$\tau_\mathrm{s}$')
plt.savefig('dens_1024_poly20_ts.png')
plt.close()

######################################################
######################################################
######################################################

fluid = 'sum-dust'
filtered = 'fluid == @fluid'
dfmono = pd.read_csv(file_dir+"/processed_data/res256_mono_ts(1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfmono.groupby(level=['ts']).transform('first'))
dfmono_0 = dfmono / t0
dfmono_0 = dfmono_0.query(filtered)

dfpoly5 = pd.read_csv(file_dir+"/processed_data/res256_poly5_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly5.groupby(level=['ts']).transform('first'))
dfpoly5_0 = dfpoly5 / t0
dfpoly5_0 = dfpoly5_0.query(filtered)

dfpoly10 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly10.groupby(level=['ts']).transform('first'))
dfpoly10_0 = dfpoly10 / t0
dfpoly10_0 = dfpoly10_0.query(filtered)

dfpoly20 = pd.read_csv(file_dir+"/processed_data/res256_poly20_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly20.groupby(level=['ts']).transform('first'))
dfpoly20_0 = dfpoly20 / t0
dfpoly20_0 = dfpoly20_0.query(filtered)

fig, ax = plt.subplots()

sns.lineplot(data=dfmono_0, x="t", y="avg99", color=medium[4], linestyle='dotted', legend=False, ax=ax)
sns.lineplot(data=dfpoly5_0, x="t", y="avg99", color=medium[0], legend=False, ax=ax)
sns.lineplot(data=dfpoly10_0, x="t", y="avg99", color=medium[1], legend=False, ax=ax)
sns.lineplot(data=dfpoly20_0, x="t", y="avg99", color=medium[2], legend=False, ax=ax)


legend_elments = [Patch(visible=False, label=r'$\mathbf{n_\mathrm{d}}$'),
                  Line2D([0], [0], color=medium[4], linestyle='dotted',  label=r'1'),
                  Line2D([0], [0], color=medium[0],  label=r'5'),
                  Line2D([0], [0], color=medium[1],  label=r'10'),
                  Line2D([0], [0], color=medium[2],  label=r'20')]
ax.legend(handles=legend_elments)
ax.set_yscale('log')
ax.set_xlim(0, 160)

ax.set_xlabel(r'$\Omega t$')
ax.set_ylabel(r'$\bar{\rho}_{\mathrm{d} \, (> \!99\%)} \, / \, \bar{\rho}_\mathrm{d}^0$')
plt.savefig('dens_ndust.png')
plt.close()

######################################################
######################################################
######################################################

fluid = 'sum-dust'
filtered = 'fluid == @fluid'
dfmono256 = pd.read_csv(file_dir+"/processed_data/res256_mono_ts(1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfmono256.groupby(level=['ts']).transform('first'))
dfmono256_0 = dfmono256 / t0
dfmono256_0 = dfmono256_0.query(filtered)

dfmono512 = pd.read_csv(file_dir+"/processed_data/res512_mono_ts(1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfmono512.groupby(level=['ts']).transform('first'))
dfmono512_0 = dfmono512 / t0
dfmono512_0 = dfmono512_0.query(filtered)

dfmono1024 = pd.read_csv(file_dir+"/processed_data/res1024_mono_ts(1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfmono1024.groupby(level=['ts']).transform('first'))
dfmono1024_0 = dfmono1024 / t0
dfmono1024_0 = dfmono1024_0.query(filtered)

dfpoly256 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly256.groupby(level=['ts']).transform('first'))
dfpoly256_0 = dfpoly256 / t0
dfpoly256_0 = dfpoly256_0.query(filtered)

dfpoly512 = pd.read_csv(file_dir+"/processed_data/res512_poly10_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly512.groupby(level=['ts']).transform('first'))
dfpoly512_0 = dfpoly512 / t0
dfpoly512_0 = dfpoly512_0.query(filtered)

dfpoly1024 = pd.read_csv(file_dir+"/processed_data/res1024_poly10_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly1024.groupby(level=['ts']).transform('first'))
dfpoly1024_0 = dfpoly1024 / t0
dfpoly1024_0 = dfpoly1024_0.query(filtered)
dfpoly1024_0

fig, ax = plt.subplots()

sns.lineplot(data=dfmono256_0, x="t", y="avg99", color=light[0], linestyle='dotted', legend=False, ax=ax)
sns.lineplot(data=dfmono512_0, x="t", y="avg99", color=light[1], linestyle='dotted', legend=False, ax=ax)
sns.lineplot(data=dfmono1024_0, x="t", y="avg99", color=light[2], linestyle='dotted', legend=False, ax=ax)

sns.lineplot(data=dfmono256_0, x="t", y="max", color=medium[0], linestyle='dotted', legend=False, ax=ax)
sns.lineplot(data=dfmono512_0, x="t",  y="max", color=medium[1], linestyle='dotted', legend=False, ax=ax)
sns.lineplot(data=dfmono1024_0, x="t",  y="max", color=medium[2], linestyle='dotted', legend=False, ax=ax)

sns.lineplot(data=dfpoly256_0, x="t", y="avg99", color=light[0], legend=False, ax=ax)
sns.lineplot(data=dfpoly512_0, x="t", y="avg99", color=light[1], legend=False, ax=ax)
sns.lineplot(data=dfpoly1024_0, x="t", y="avg99", color=light[2], legend=False, ax=ax)

sns.lineplot(data=dfpoly256_0, x="t",  y="max", color=medium[0], legend=False, ax=ax)
sns.lineplot(data=dfpoly512_0, x="t",  y="max", color=medium[1], legend=False, ax=ax)
sns.lineplot(data=dfpoly1024_0, x="t",  y="max", color=medium[2], legend=False, ax=ax)

legend_elments = [Patch(visible=False, label=r'$\mathbf{PC}$'),
                  Line2D([0], [0], color=medium[5],  label=r'$\rho_{\mathrm{max}}$'),
                  Line2D([0], [0], color=light[5],  label=r'$\bar{\rho}_{>99\%}$'),
                  Patch(visible=False, label=r'$\mathbf{Fluid}$'),
                  Line2D([0], [0], color=colortxt, linestyle='dotted',  label=r'mSI'),
                  Line2D([0], [0], color=colortxt,  label=r'sum PSI'),
                  Patch(visible=False, label=r'$\mathbf{N_\mathrm{grid}}$'),
                  Line2D([0], [0], color=medium[0],  label=r'256'),
                  Line2D([0], [0], color=medium[1],  label=r'512'),
                  Line2D([0], [0], color=medium[2],  label=r'1024')]
ax.legend(handles=legend_elments, loc='upper left')

ax.set_yscale('log')
ax.set_xlim(0, 160)
ax.set_xlabel(r'$\Omega t$')
ax.set_ylabel(r'$\bar{\rho}_{\mathrm{d} \, (> \mathrm{PC})} \, / \, \bar{\rho}_\mathrm{d}^0$')
plt.savefig('dens_res.png')
plt.close()

######################################################
######################################################
######################################################

fluid = 'sum-dust'
filtered = 'fluid == @fluid'
dfmono_a0 = pd.read_csv(file_dir+"/processed_data/res256_mono_ts(1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfmono_a0.groupby(level=['ts']).transform('first'))
dfmono_a0_0 = dfmono_a0 / t0
dfmono_a0_0 = dfmono_a0_0.query(filtered)

dfmono_a8 = pd.read_csv(file_dir+"/processed_data/res256_mono_ts(1e-1)_k30_alpha8_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfmono_a8.groupby(level=['ts']).transform('first'))
dfmono_a8_0 = dfmono_a8 / t0
dfmono_a8_0 = dfmono_a8_0.query(filtered)

dfmono_a7 = pd.read_csv(file_dir+"/processed_data/res256_mono_ts(1e-1)_k30_alpha7_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfmono_a7.groupby(level=['ts']).transform('first'))
dfmono_a7_0 = dfmono_a7 / t0
dfmono_a7_0 = dfmono_a7_0.query(filtered)

dfmono_a6 = pd.read_csv(file_dir+"/processed_data/res256_mono_ts(1e-1)_k30_alpha6_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfmono_a6.groupby(level=['ts']).transform('first'))
dfmono_a6_0 = dfmono_a6 / t0
dfmono_a6_0 = dfmono_a6_0.query(filtered)

dfpoly_a0 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly_a0.groupby(level=['ts']).transform('first'))
dfpoly_a0_0 = dfpoly_a0 / t0
dfpoly_a0_0 = dfpoly_a0_0.query(filtered)

dfpoly_a8 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_alpha8_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly_a8.groupby(level=['ts']).transform('first'))
dfpoly_a8_0 = dfpoly_a8 / t0
dfpoly_a8_0 = dfpoly_a8_0.query(filtered)

dfpoly_a7 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_alpha7_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly_a7.groupby(level=['ts']).transform('first'))
dfpoly_a7_0 = dfpoly_a7 / t0
dfpoly_a7_0 = dfpoly_a7_0.query(filtered)

dfpoly_a6 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_alpha6_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly_a6.groupby(level=['ts']).transform('first'))
dfpoly_a6_0 = dfpoly_a6 / t0
dfpoly_a6_0 = dfpoly_a6_0.query(filtered)

fig =  plt.figure(figsize=(3.54331, 3.54331))
gs_main = gridspec.GridSpec(2,1, figure=fig, height_ratios=[1, 1], hspace=0)

ax0 = fig.add_subplot(gs_main[0])

sns.lineplot(data=dfmono_a0_0, x="t", y="ampfft", color=medium[0], linestyle='dotted', legend=False, ax=ax0)
sns.lineplot(data=dfmono_a8_0, x="t", y="ampfft", color=medium[1], linestyle='dotted', legend=False, ax=ax0)
sns.lineplot(data=dfmono_a7_0, x="t", y="ampfft", color=medium[2], linestyle='dotted', legend=False, ax=ax0)
sns.lineplot(data=dfmono_a6_0, x="t", y="ampfft", color=medium[4], linestyle='dotted', legend=False, ax=ax0)

sns.lineplot(data=dfpoly_a0_0, x="t", y="ampfft", color=medium[0], legend=False, ax=ax0)
sns.lineplot(data=dfpoly_a8_0, x="t", y="ampfft", color=medium[1], legend=False, ax=ax0)
sns.lineplot(data=dfpoly_a7_0, x="t", y="ampfft", color=medium[2], legend=False, ax=ax0)
sns.lineplot(data=dfpoly_a6_0, x="t", y="ampfft", color=medium[4], legend=False, ax=ax0)

t = np.linspace(0,350, 100)
ax0.plot(t, 1*np.exp(ts_e3_e1_growth*t), color=dark[0], linewidth=lwidth*.8, linestyle='dashdot')
ax0.plot(t, 1*np.exp(ts_e3_e1_growth_alpha8*t), color=dark[1], linewidth=lwidth*.8, linestyle='dashdot')
ax0.plot(t, 1*np.exp(ts_e3_e1_growth_alpha7*t), color=dark[2], linewidth=lwidth*.8, linestyle='dashdot')
ax0.plot(t, 1*np.exp(ts_e3_e1_growth_alpha6*t), color=dark[4], linewidth=lwidth*.8, linestyle='dashdot')

ax0.plot(t[t<40], 1*np.exp(ts_e1_growth*t[t<40]), color=dark[0], linewidth=lwidth*.8, linestyle='dashed')
ax0.plot(t[t<50], 1*np.exp(ts_e1_growth_alpha8*t[t<50]), color=dark[1], linewidth=lwidth*.8, linestyle='dashed')
ax0.plot(t[t<60], 1*np.exp(ts_e1_growth_alpha7*t[t<60]), color=dark[2], linewidth=lwidth*.8, linestyle='dashed')
ax0.plot(t[t<110], 1*np.exp(ts_e1_growth_alpha6*t[t<110]), color=dark[4], linewidth=lwidth*.8, linestyle='dashed')

legend_elments = [Line2D([0], [0], color=colortxt, linestyle='dotted',  label=r'mSI'),
                  Line2D([0], [0], color=colortxt,  label=r'sum PSI'),
                  Patch(visible=False, label=r'$\mathbf{Diffusion}$'),
                  Line2D([0], [0], color=medium[0],  label=r'$\alpha = 0$'),
                  Line2D([0], [0], color=medium[1],  label=r'$\alpha = 10^{-8}$'),
                  Line2D([0], [0], color=medium[2],  label=r'$\alpha = 10^{-7}$'),
                  Line2D([0], [0], color=medium[4],  label=r'$\alpha = 10^{-6}$')]

legend2 = plt.legend(handles = [Patch(visible=False, label=r'$\mathbf{\propto \exp(\Im{(\omega)\cdot t})}$'),
                                Line2D([0], [0], color=colortxt, linestyle='dashed', lw=1, label=r'mSI'),
                                Line2D([0], [0], color=colortxt, linestyle='dashdot', lw=1, label=r'PSI')],loc='upper right')
plt.gca().add_artist(legend2)
ax0.legend(handles=legend_elments, loc='lower right')


plt.setp(ax0.get_xticklabels(), visible=False)

ax0.set_ylabel(r'$P_K/P^0_K$')
ax0.set_yscale('log')
ax0.set_xlim(0, 500)
ax0.set_ylim(1e-1, 1e6)

ax1 = fig.add_subplot(gs_main[1])

sns.lineplot(data=dfmono_a0_0, x="t",  y="avg99", color=medium[0], linestyle='dotted', legend=False, ax=ax1, alpha=.8)
sns.lineplot(data=dfmono_a8_0, x="t",  y="avg99", color=medium[1], linestyle='dotted', legend=False, ax=ax1, alpha=.8)
sns.lineplot(data=dfmono_a7_0, x="t",  y="avg99", color=medium[2], linestyle='dotted', legend=False, ax=ax1, alpha=.8)
sns.lineplot(data=dfmono_a6_0, x="t",  y="avg99", color=medium[4], linestyle='dotted', legend=False, ax=ax1, alpha=.8)

sns.lineplot(data=dfpoly_a0_0, x="t",  y="avg99", color=medium[0], legend=False, ax=ax1)
sns.lineplot(data=dfpoly_a8_0, x="t",  y="avg99", color=medium[1], legend=False, ax=ax1)
sns.lineplot(data=dfpoly_a7_0, x="t",  y="avg99", color=medium[2], legend=False, ax=ax1)
sns.lineplot(data=dfpoly_a6_0, x="t",  y="avg99", color=medium[4], legend=False, ax=ax1)

ax1.set_yscale('log')
ax1.set_xlim(0, 500)
ax0.set_ylabel(r'$P_K/P^0_K$')
ax1.set_xlabel(r'$\Omega t$')
ax1.set_ylabel(r'$\bar{\rho}_{\mathrm{d} \, (> \!99\%)} \, / \, \bar{\rho}_\mathrm{d}^0$')
plt.savefig('dens_amp_alpha.png')
plt.close()

######################################################
######################################################
######################################################

fluid = 'sum-dust'
filtered = 'fluid == @fluid'
filtered2 = '0 < t'

dfpolym05_mono= pd.read_csv(file_dir+"/processed_data/res256_mono_ts(1e-1)_WN-4_mu05_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
dfpolym05_mono = dfpolym05_mono.query(filtered)
dfpolym05_mono = dfpolym05_mono.query(filtered2)
t0 = (dfpolym05_mono.groupby(level=['ts']).transform('first'))
dfpolym05_mono_0 = dfpolym05_mono / t0

dfpolym1_mono= pd.read_csv(file_dir+"/processed_data/res256_mono_ts(1e-1)_WN-4_mu1_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
dfpolym1_mono = dfpolym1_mono.query(filtered)
dfpolym1_mono = dfpolym1_mono.query(filtered2)
t0 = (dfpolym1_mono.groupby(level=['ts']).transform('first'))
dfpolym1_mono_0 = dfpolym1_mono/ t0

dfpolym3_mono= pd.read_csv(file_dir+"/processed_data/res256_mono_ts(1e-1)_WN-4_mu3_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
dfpolym3_mono = dfpolym3_mono.query(filtered)
dfpolym3_mono = dfpolym3_mono.query(filtered2)
t0 = (dfpolym3_mono.groupby(level=['ts']).transform('first'))
dfpolym3_mono_0 = dfpolym3_mono/ t0

dfpolym05= pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_WN-4_mu05_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
dfpolym05 = dfpolym05.query(filtered)
dfpolym05 = dfpolym05.query(filtered2)
t0 = (dfpolym05.groupby(level=['ts']).transform('first'))
dfpolym05_0 = dfpolym05 / t0
dfpolym05_0

dfpolym1= pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_WN-4_mu1_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
dfpolym1 = dfpolym1.query(filtered)
dfpolym1 = dfpolym1.query(filtered2)
t0 = (dfpolym1.groupby(level=['ts']).transform('first'))
dfpolym1_0 = dfpolym1/ t0
dfpolym1_0

dfpolym3= pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_WN-4_mu3_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
dfpolym3 = dfpolym3.query(filtered)
dfpolym3 = dfpolym3.query(filtered2)
t0 = (dfpolym3.groupby(level=['ts']).transform('first'))
dfpolym3_0 = dfpolym3/ t0

fig, ax = plt.subplots()

sns.lineplot(data=dfpolym05_mono_0, x="t",  y="avg99", color=medium[0], legend=False, ax=ax, linestyle='dotted')
sns.lineplot(data=dfpolym1_mono_0, x="t",  y="avg99", color=medium[1], legend=False, ax=ax, linestyle='dotted')
sns.lineplot(data=dfpolym3_mono_0, x="t",  y="avg99", color=medium[2], legend=False, ax=ax, linestyle='dotted')

sns.lineplot(data=dfpolym05_0, x="t",  y="avg99", color=medium[0], legend=False, ax=ax)
sns.lineplot(data=dfpolym1_0, x="t",  y="avg99", color=medium[1], legend=False, ax=ax)
sns.lineplot(data=dfpolym3_0, x="t",  y="avg99", color=medium[2], legend=False, ax=ax)

legend_elments = [Line2D([0], [0], color=colortxt, linestyle='dotted',  label=r'mSI'),
                  Line2D([0], [0], color=colortxt,  label=r'PSI'),
                  Patch(visible=False, label=r'$\mathbf{Dust\!/\!Gas \, Ratio}$'),
                  Line2D([0], [0], color=medium[0],  label=r'$\mu = 0.5$'),
                  Line2D([0], [0], color=medium[1],  label=r'$\mu = 1$'),
                  Line2D([0], [0], color=medium[2],  label=r'$\mu = 3$')]
ax.legend(handles=legend_elments)

ax.set_xlabel(r'$\Omega t$')
ax.set_ylabel(r'$\bar{\rho}_{(> \!99\%)} \, / \, \bar{\rho}^0_{(> \!99\%)}$')
ax.set_yscale('log')
ax.set_xlim(0, 500)

ax.set_xlabel(r'$\Omega t$')
ax.set_ylabel(r'$\bar{\rho}_{\mathrm{d} \, (> \!99\%)} \, / \, \bar{\rho}_\mathrm{d}^0$')
plt.savefig('dens_mu_WN4.png')
plt.close()

######################################################
######################################################
######################################################

fluid = 'sum-dust'
filtered = 'fluid == @fluid'
dfmono_ts01 = pd.read_csv(file_dir+"/processed_data/res256_mono_ts(1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfmono_ts01.groupby(level=['ts']).transform('first'))
dfmono_ts01_0 = dfmono_ts01 / t0
dfmono_ts01_0 = dfmono_ts01_0.query(filtered)

dfmono_ts005 = pd.read_csv(file_dir+"/processed_data/res256_mono_ts(5e-2)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfmono_ts005.groupby(level=['ts']).transform('first'))
dfmono_ts005_0 = dfmono_ts005 / t0
dfmono_ts005_0 = dfmono_ts005_0.query(filtered)

dfmono_ts02 = pd.read_csv(file_dir+"/processed_data/res256_mono_ts(2e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfmono_ts02.groupby(level=['ts']).transform('first'))
dfmono_ts02_0 = dfmono_ts02 / t0
dfmono_ts02_0 = dfmono_ts02_0.query(filtered)

dfe3 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfe3.groupby(level=['ts']).transform('first'))
dfe3_0 = dfe3 / t0
dfe3_0 = dfe3_0.query(filtered)

dfe4 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-4|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfe4.groupby(level=['ts']).transform('first'))
dfe4_0 = dfe4 / t0
dfe4_0 = dfe4_0.query(filtered)

df5 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|5e-2)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (df5.groupby(level=['ts']).transform('first'))
df5_0 = df5 / t0
df5_0 = df5_0.query(filtered)

df2 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|2e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (df2.groupby(level=['ts']).transform('first'))
df2_0 = df2 / t0
df2_0 = df2_0.query(filtered)

fig =  plt.figure(figsize=(3.54331, 3.54331))
gs_main = gridspec.GridSpec(2,1, figure=fig, height_ratios=[1, 1], hspace=0)

ax0 = fig.add_subplot(gs_main[0])


sns.lineplot(data=dfmono_ts005_0, x="t", y="ampfft", color=medium[0], linestyle='dotted', legend=False, ax=ax0)
sns.lineplot(data=dfmono_ts01_0, x="t", y="ampfft", color=medium[1], linestyle='dotted', legend=False, ax=ax0)
sns.lineplot(data=dfmono_ts02_0, x="t", y="ampfft", color=medium[2], linestyle='dotted', legend=False, ax=ax0)

sns.lineplot(data=df5_0, x="t", y="ampfft", color=medium[0], legend=False, ax=ax0)
sns.lineplot(data=dfe3_0, x="t", y="ampfft", color=medium[1], legend=False, ax=ax0)
sns.lineplot(data=df2_0, x="t", y="ampfft", color=medium[2], legend=False, ax=ax0)

t = np.linspace(0,160, 100)
ax0.plot(t, 1*np.exp(ts_e3_5e2_growth*t), color=dark[0], linewidth=lwidth*.8, linestyle='dashdot')
ax0.plot(t[t<200], 1*np.exp(ts_e3_e1_growth*t[t<200]), color=dark[1], linewidth=lwidth*.8, linestyle='dashdot')
ax0.plot(t, 1*np.exp(ts_e3_2e1_growth*t), color=dark[2], linewidth=lwidth*.8, linestyle='dashdot')

ax0.plot(t[t<35], 1*np.exp(ts_5e2_growth*t[t<35]), color=dark[0], linewidth=lwidth*.8, linestyle='dashed')
ax0.plot(t[t<35], 1*np.exp(ts_e1_growth*t[t<35]), color=dark[1], linewidth=lwidth*.8, linestyle='dashed')
ax0.plot(t[t<35], 1*np.exp(ts_2e1_growth*t[t<35]), color=dark[2], linewidth=lwidth*.8, linestyle='dashed')

legend_elments = [Line2D([0], [0], color=colortxt, linestyle='dotted',  label=r'mSI'),
                  Line2D([0], [0], color=colortxt,  label=r'PSI'),
                  Patch(visible=False, label=r'$\mathbf{\tau_s \ max}$'),
                  Line2D([0], [0], color=medium[0],  label=r'$\tau_s=0.05$'),
                  Line2D([0], [0], color=medium[1],  label=r'$\tau_s=0.1$'),
                  Line2D([0], [0], color=medium[2],  label=r'$\tau_s=0.2$'),]

legend2 = plt.legend(handles = [Patch(visible=False, label=r'$\mathbf{\propto \exp(\Im{(\omega)\cdot t})}$'),
                                Line2D([0], [0], color=colortxt, linestyle='dashed', lw=1, label=r'mSI'),
                                Line2D([0], [0], color=colortxt, linestyle='dashdot', lw=1, label=r'PSI')],loc='upper right')
plt.gca().add_artist(legend2)
ax0.legend(handles=legend_elments, loc='lower right')


plt.setp(ax0.get_xticklabels(), visible=False)

ax0.set_ylabel(r'$P_K/P^0_K$')
ax0.set_yscale('log')
ax0.set_xlim(0, 160)
ax0.set_ylim(1e-1, 1e6)

ax1 = fig.add_subplot(gs_main[1])

sns.lineplot(data=dfmono_ts005_0, x="t",  y="avg99", color=medium[0], linestyle='dotted', legend=False, ax=ax1)
sns.lineplot(data=dfmono_ts01_0, x="t",  y="avg99", color=medium[1], linestyle='dotted', legend=False, ax=ax1)
sns.lineplot(data=dfmono_ts02_0, x="t",  y="avg99", color=medium[2], linestyle='dotted', legend=False, ax=ax1)


sns.lineplot(data=df5_0, x="t",  y="avg99", color=medium[0], legend=False, ax=ax1)
sns.lineplot(data=dfe3_0, x="t",  y="avg99", color=medium[1], legend=False, ax=ax1)
sns.lineplot(data=df2_0, x="t",  y="avg99", color=medium[2], legend=False, ax=ax1)

ax1.set_yscale('log')
ax1.set_xlim(0, 160)
ax0.set_ylabel(r'$P_K/P^0_K$')
ax1.set_xlabel(r'$\Omega t$')
ax1.set_ylabel(r'$\bar{\rho}_{\mathrm{d} \, (> \!99\%)} \, / \, \bar{\rho}_\mathrm{d}^0$')
plt.savefig('dens_amp_tspeak.png')
plt.close()

######################################################
######################################################
######################################################

fluid = 'sum-dust'
filtered = 'fluid == @fluid'
dfpoly10 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly10.groupby(level=['ts']).transform('first'))
dfpoly10_0 = dfpoly10 / t0
dfpoly10_0 = dfpoly10_0.query(filtered)

dfpoly20 = pd.read_csv(file_dir+"/processed_data/res256_poly20_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly20.groupby(level=['ts']).transform('first'))
dfpoly20_0 = dfpoly20 / t0
dfpoly20_0 = dfpoly20_0.query(filtered)

dfpoly10disc = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_disc_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly10disc.groupby(level=['ts']).transform('first'))
dfpoly100 = dfpoly10disc / t0
dfpoly100 = dfpoly100.query(filtered)

dfpoly20disc = pd.read_csv(file_dir+"/processed_data/res256_poly20_ts(1e-3|1e-1)_k30_disc_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly20disc.groupby(level=['ts']).transform('first'))
dfpoly200 = dfpoly20disc / t0
dfpoly200 = dfpoly200.query(filtered)

dfpoly40disc = pd.read_csv(file_dir+"/processed_data/res256_poly40_ts(1e-3|1e-1)_k30_disc_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly40disc.groupby(level=['ts']).transform('first'))
dfpoly400 = dfpoly40disc / t0
dfpoly400 = dfpoly400.query(filtered)

fig, ax = plt.subplots()

sns.lineplot(data=dfpoly100, x="t",  y="ampfft", color=medium[0], linestyle='dotted', legend=False, ax=ax)
sns.lineplot(data=dfpoly200, x="t",  y="ampfft", color=medium[1], linestyle='dotted',legend=False, ax=ax)
sns.lineplot(data=dfpoly10_0, x="t",  y="ampfft", color=medium[0], legend=False, ax=ax)
sns.lineplot(data=dfpoly20_0, x="t",  y="ampfft", color=medium[1], legend=False, ax=ax)
sns.lineplot(data=dfpoly400, x="t",  y="ampfft", color=medium[2], linewidth=lwidth*2, linestyle='dotted',legend=False, ax=ax)


plt.plot(t[t< 150], 1*np.exp(ts_e3_e1_growth*t[t < 150]), color=colortxt, linewidth=lwidth*.8, linestyle='dashdot')

legend_elments = [Patch(visible=False, label=r'$\mathbf{Distribution}$'),
                  Line2D([0], [0], color=colortxt, linestyle='dotted',  label=r'Discrete'),
                  Line2D([0], [0], color=colortxt,  label=r'Gauss-Legendre'),
                  Patch(visible=False, label=r'$\mathbf{n_\mathrm{d}}$'),
                  Line2D([0], [0], color=medium[0],  label=r'$10$'),
                  Line2D([0], [0], color=medium[1],  label=r'$20$'),
                  Line2D([0], [0], color=medium[2],  label=r'40')]
legend2 = plt.legend(handles = [Patch(visible=False, label=r'$\mathbf{Analytical}$'),
                                Line2D([0], [0], color=colortxt, linestyle='dashdot', lw=1, label=r'$\mathbf{\propto \exp(\Im{(\omega)\cdot t})}$')],loc='lower right')
plt.gca().add_artist(legend2)

ax.legend(handles=legend_elments, loc='upper left')

ax.set_yscale('log')
ax.set_xlim(0, 160)
ax.set_ylim(1e0,1e4)
ax.set_xlabel(r'$\Omega t$')
ax.set_ylabel(r'$P_K$')
plt.savefig('Amp_discrete_256_ts.png')
plt.close()

######################################################
######################################################
######################################################

fluid = 'sum-dust'
filtered = 'fluid == @fluid'
dfpoly_mrn32 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_mrn32_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly_mrn32.groupby(level=['ts']).transform('first'))
dfpoly_mrn32_0 = dfpoly_mrn32 / t0
dfpoly_mrn32_0 = dfpoly_mrn32_0.query(filtered)

dfpoly_mrn35 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly_mrn35.groupby(level=['ts']).transform('first'))
dfpoly_mrn35_0 = dfpoly_mrn35 / t0
dfpoly_mrn35_0 = dfpoly_mrn35_0.query(filtered)

dfpoly_mrn38 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_mrn38_dens_amp.csv", index_col=['t','fluid', 'ts'])
# normalize values by first timestep
t0 = (dfpoly_mrn38.groupby(level=['ts']).transform('first'))
dfpoly_mrn38_0 = dfpoly_mrn38 / t0
dfpoly_mrn38_0 = dfpoly_mrn38_0.query(filtered)

fig, ax = plt.subplots()

sns.lineplot(data=dfpoly_mrn38_0, x="t",  y="avg99", color=medium[0], legend=False, ax=ax)
sns.lineplot(data=dfpoly_mrn35_0, x="t",  y="avg99", color=medium[1], legend=False, ax=ax)
sns.lineplot(data=dfpoly_mrn32_0, x="t",  y="avg99", color=medium[2], legend=False, ax=ax)

legend_elments = [Line2D([0], [0], color=medium[0],  label=r'$\sigma^0_{(\tau_s)} \propto \tau_s^{-1/5}$'),
                  Line2D([0], [0], color=medium[1],  label=r'$\sigma^0_{(\tau_s)} \propto \tau_s^{-1/2}$'),
                  Line2D([0], [0], color=medium[2],  label=r'$\sigma^0_{(\tau_s)} \propto \tau_s^{-4/5}$')]
ax.legend(handles=legend_elments)

# ax.set_title(r'Sum-dust species')
ax.set_yscale('log')
ax.set_xlim(0, 200)
ax.set_xlabel(r'$\Omega t$')
ax.set_ylabel(r'$\bar{\rho}_{\mathrm{d} \, (> \!99\%)} \, / \, \bar{\rho}_\mathrm{d}^0$')
plt.savefig('dens_slope.png')
plt.close()

######################################################
######################################################
######################################################

filtered = '150 <= t <= 250' 
df = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_dens_ts_cont.csv", index_col=['t','ts'])
df["avg68/ts_0"] = df["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df.index.get_level_values('ts')))
df["avg90/ts_0"] = df["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df.index.get_level_values('ts')))
df["avg95/ts_0"] = df["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df.index.get_level_values('ts')))
df["avg99/ts_0"] = df["avg99/ts"]/(sigma_ts0(1e-3, 1e-1,df.index.get_level_values('ts')))
df_t = df.query(filtered)

dfe4 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-4|1e-1)_k30_dens_ts_cont.csv", index_col=['t','ts'])
dfe4["avg68/ts_0"] = dfe4["avg68/ts"]/(sigma_ts0(1e-4, 1e-1, dfe4.index.get_level_values('ts')))
dfe4["avg90/ts_0"] = dfe4["avg90/ts"]/(sigma_ts0(1e-4, 1e-1, dfe4.index.get_level_values('ts')))
dfe4["avg95/ts_0"] = dfe4["avg95/ts"]/(sigma_ts0(1e-4, 1e-1, dfe4.index.get_level_values('ts')))
dfe4["avg99/ts_0"] = dfe4["avg99/ts"]/(sigma_ts0(1e-4, 1e-1,dfe4.index.get_level_values('ts')))
df_te4 = dfe4.query(filtered)

df5 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|5e-2)_k30_dens_ts_cont.csv", index_col=['t','ts'])
df5["avg68/ts_0"] = df5["avg68/ts"]/(sigma_ts0(1e-3, 5e-2, df5.index.get_level_values('ts')))
df5["avg90/ts_0"] = df5["avg90/ts"]/(sigma_ts0(1e-3, 5e-2, df5.index.get_level_values('ts')))
df5["avg95/ts_0"] = df5["avg95/ts"]/(sigma_ts0(1e-3, 5e-2, df5.index.get_level_values('ts')))
df5["avg99/ts_0"] = df5["avg99/ts"]/(sigma_ts0(1e-3, 5e-2, df5.index.get_level_values('ts')))
df_t5 = df5.query(filtered)

df2 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|2e-1)_k30_dens_ts_cont.csv", index_col=['t','ts'])
df2["avg68/ts_0"] = df2["avg68/ts"]/(sigma_ts0(1e-3, 2e-1, df2.index.get_level_values('ts')))
df2["avg90/ts_0"] = df2["avg90/ts"]/(sigma_ts0(1e-3, 2e-1, df2.index.get_level_values('ts')))
df2["avg95/ts_0"] = df2["avg95/ts"]/(sigma_ts0(1e-3, 2e-1, df2.index.get_level_values('ts')))
df2["avg99/ts_0"] = df2["avg99/ts"]/(sigma_ts0(1e-3, 2e-1, df2.index.get_level_values('ts')))

df_t2 = df2.query(filtered)

ymin = 1
ymax = 2e1
fig, ax = plt.subplots()
paired = sns.color_palette("Paired")
colours = sns.color_palette("tab10")

x = np.logspace(-3, np.log10(2e-1))
plt.plot(x, np.ones_like(x), color=colortxt, linestyle='dotted')

plt.vlines(ts_e3_e1_vres, ymin=ymin, ymax=ymax, linestyles='dashed', color=medium[0])
plt.vlines(ts_e3_5e2_vres, ymin=ymin, ymax=ymax, linestyles='dashed', color=medium[1])
plt.vlines(ts_e3_2e1_vres, ymin=ymin, ymax=ymax, linestyles='dashed', color=medium[2])

sns.lineplot(data=df_t, x="ts", y="avg99/ts_0", color=medium[0], ax=ax, errorbar=('ci', 99.7))
sns.lineplot(data=df_t5, x="ts", y="avg99/ts_0", color=medium[1],  ax=ax, errorbar=('ci', 99.7))
sns.lineplot(data=df_t2, x="ts", y="avg99/ts_0", color=medium[2],  ax=ax, errorbar=('ci', 99.7))
       
legend_elments = [Patch(visible=False, label=r'$\mathbf{\tau_{s,\ max}}$'),
                  Line2D([0], [0], color=medium[0],  label=r'$10^{-1}$'),
                  Line2D([0], [0], color=medium[1],  label=r'$5 \! \cdot \! 10^{-2}$'),
                  Line2D([0], [0], color=medium[2],  label=r'$2 \! \cdot \! 10^{-1}$'),
                  Patch(visible=False, label=r'$\mathbf{Analytical}$'),
                  Line2D([0], [0], color=colortxt, linestyle='dashed', label=r'$\tau_s \text{ at } v_{\text{res}}$')]
ax.legend(handles=legend_elments[:])

# Add the legend manually to the Axes.
ax.set_xlabel(r'$\tau_{s}$')
ax.set_ylabel(r'$[\bar{\rho}_{>99\%}/\tau_s]/\sigma^0_{(\tau_s)}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-3, 2e-1)
ax.set_ylim(ymin, ymax)

ax.text(.5, 0.92, r'$150 \leq \Omega t \leq 250$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

ax.set_xlabel(r'$\tau_\mathrm{s}$')
ax.set_ylabel(r'$\bar{\sigma}_{(\rho > 99\%)}/\sigma^0$')
plt.savefig('normsize_peak.png')
plt.close()

######################################################
######################################################
######################################################

filtered = '135 <= t <= 145'  

df5 = pd.read_csv(file_dir+"/processed_data/res1024_poly5_ts(1e-3|1e-1)_k30_dens_ts_cont.csv", index_col=['t','ts'])
df5["avg68/ts_0"] = df5["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df5.index.get_level_values('ts')))
df5["avg90/ts_0"] = df5["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df5.index.get_level_values('ts')))
df5["avg95/ts_0"] = df5["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df5.index.get_level_values('ts')))
df5["avg99/ts_0"] = df5["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df5.index.get_level_values('ts')))
df_t5 = df5.query(filtered)

df10 = pd.read_csv(file_dir+"/processed_data/res1024_poly10_ts(1e-3|1e-1)_k30_dens_ts_cont.csv", index_col=['t','ts'])
df10["avg68/ts_0"] = df10["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df10.index.get_level_values('ts')))
df10["avg90/ts_0"] = df10["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df10.index.get_level_values('ts')))
df10["avg95/ts_0"] = df10["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df10.index.get_level_values('ts')))
df10["avg99/ts_0"] = df10["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df10.index.get_level_values('ts')))
df_t10 = df10.query(filtered)

df20 = pd.read_csv(file_dir+"/processed_data/res1024_poly20_ts(1e-3|1e-1)_k30_dens_ts_cont.csv", index_col=['t','ts'])
df20["avg68/ts_0"] = df20["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df20.index.get_level_values('ts')))
df20["avg90/ts_0"] = df20["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df20.index.get_level_values('ts')))
df20["avg95/ts_0"] = df20["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df20.index.get_level_values('ts')))
df20["avg99/ts_0"] = df20["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df20.index.get_level_values('ts')))
df_t20 = df20.query(filtered)

fig, ax = plt.subplots()
ymin = 1
ymax = 2e1
plt.plot(x, np.ones_like(x), color=colortxt, linestyle='dotted')
plt.vlines(ts_e3_e1_vres, ymin=ymin, ymax=ymax, linestyles='dashed', color=colortxt)
plt.vlines(ts_e3_e1_avg0, ymin=ymin, ymax=ymax, linestyles='dashdot', color=colortxt)

sns.lineplot(data=df_t5, x="ts", y="avg99/ts_0", ax=ax, errorbar=('ci', 99.7), color=medium[0])
sns.lineplot(data=df_t10, x="ts", y="avg99/ts_0", ax=ax, errorbar=('ci', 99.7), color=medium[1])
sns.lineplot(data=df_t20, x="ts", y="avg99/ts_0", ax=ax, errorbar=('ci', 99.7), color=medium[2])
          
legend_elments = [Patch(visible=False, label=r'$\mathbf{n_\text{d}}$'),
                  Line2D([0], [0], color=medium[0],  label=r'$5$'),
                  Line2D([0], [0], color=medium[1],  label=r'$10$'),
                  Line2D([0], [0], color=medium[2],  label=r'$20$'),
                  Patch(visible=False, label=r'$\mathbf{Analytical}$'),
                  Line2D([0], [0], color=colortxt, linestyle='dashed', label=r'$\tau_s \text{ at } v_{\text{res}}$'),
                  Line2D([0], [0], color=colortxt, linestyle='dashdot', label=r'$\bar{\tau}_s^0$')
                  ] 

ax.legend(handles=legend_elments[:], loc='upper left')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-3, 1e-1)
ax.set_ylim(ymin, ymax)
ax.text(.5, 0.92, r'$135 \leq \Omega t \leq 145$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

ax.set_xlabel(r'$\tau_\mathrm{s}$')
ax.set_ylabel(r'$\bar{\sigma}_{(\rho > 99\%)}/\sigma^0$')
plt.savefig('normsize_ndust.png')
plt.close()

######################################################
######################################################
######################################################

filtered = '140 <= t <= 160'

df256 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_dens_ts_cont.csv", index_col=['t', 'ts'])
df256["avg68/ts_0"] = df256["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df256.index.get_level_values('ts')))
df256["avg90/ts_0"] = df256["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df256.index.get_level_values('ts')))
df256["avg95/ts_0"] = df256["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df256.index.get_level_values('ts')))
df256["avg99/ts_0"] = df256["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df256.index.get_level_values('ts')))
df_t256 = df256.query(filtered)


df512 = pd.read_csv(file_dir+"/processed_data/res512_poly10_ts(1e-3|1e-1)_k30_dens_ts_cont.csv", index_col=['t', 'ts'])
df512["avg68/ts_0"] = df512["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df512.index.get_level_values('ts')))
df512["avg90/ts_0"] = df512["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df512.index.get_level_values('ts')))
df512["avg95/ts_0"] = df512["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df512.index.get_level_values('ts')))
df512["avg99/ts_0"] = df512["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df512.index.get_level_values('ts')))
df_t512 = df512.query(filtered)

df1024 = pd.read_csv(file_dir+"/processed_data/res1024_poly10_ts(1e-3|1e-1)_k30_dens_ts_cont.csv", index_col=['t', 'ts'])
df1024["avg68/ts_0"] = df1024["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df1024.index.get_level_values('ts')))
df1024["avg90/ts_0"] = df1024["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df1024.index.get_level_values('ts')))
df1024["avg95/ts_0"] = df1024["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df1024.index.get_level_values('ts')))
df1024["avg99/ts_0"] = df1024["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df1024.index.get_level_values('ts')))
df_t1024 = df1024.query(filtered)


x = np.logspace(-3, -1, 1000)
fig, ax = plt.subplots()
plt.plot(x, np.ones_like(x), color=colortxt, linestyle='dotted')
plt.vlines(ts_e3_e1_vres, ymin=ymin, ymax=ymax, linestyles='dashed', color=colortxt)
plt.vlines(ts_e3_e1_avg0, ymin=ymin, ymax=ymax, linestyles='dashdot', color=colortxt)

sns.lineplot(data=df_t256, x="ts", y="avg90/ts_0", ax=ax, errorbar=('ci', 99.7), color=dark[0])
sns.lineplot(data=df_t256, x="ts", y="avg99/ts_0", ax=ax, errorbar=('ci', 99.7), color=medium[0])

sns.lineplot(data=df_t512, x="ts", y="avg90/ts_0", ax=ax, errorbar=('ci', 99.7), color=light[1])
sns.lineplot(data=df_t512, x="ts", y="avg99/ts_0", ax=ax, errorbar=('ci', 99.7), color=medium[1])

sns.lineplot(data=df_t1024, x="ts", y="avg90/ts_0", ax=ax, errorbar=('ci', 99.7), color=light[2])
sns.lineplot(data=df_t1024, x="ts", y="avg99/ts_0", ax=ax, errorbar=('ci', 99.7), color=medium[2])
        
legend_elments = [Line2D([0], [0], color=medium[5],  label=r'$\bar{\rho}_{>99\%}$'),
                  Line2D([0], [0], color=light[5],  label=r'$\bar{\rho}_{>90\%}$'),
                  Patch(visible=False, label=r'$\mathbf{N_{grid}}$'),
                  Line2D([0], [0], color=medium[0],  label=r'$256$'),
                  Line2D([0], [0], color=medium[1],  label=r'$512$'),
                  Line2D([0], [0], color=medium[2],  label=r'$1024$'),
                  Patch(visible=False, label=r'$\mathbf{Analytical}$'),
                  Line2D([0], [0], color=colortxt, linestyle='dashed', label=r'$\tau_s \text{ at } v_{\text{res}}$'),
                  Line2D([0], [0], color=colortxt, linestyle='dashdot', label=r'$\bar{\tau}^0_s$')] 
ax.legend(handles=legend_elments[:], loc='upper left')


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(ymin, ymax)
ax.set_xlim(1e-3,1e-1)

ax.text(.5, 0.92, r'$140 \leq \Omega t \leq 160$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

ax.set_xlabel(r'$\tau_\mathrm{s}$')
ax.set_ylabel(r'$\bar{\sigma}_{(\rho > \mathrm{PC})}/\sigma^0$')
plt.savefig('normsize_res.png')

######################################################
######################################################
######################################################

filtered = '0 <= t <= 160'
df1024 = pd.read_csv(file_dir+"/processed_data/res1024_poly10_ts(1e-3|1e-1)_k30_dens_ts_cont.csv", index_col=['t', 'ts'])
df1024["avg68/ts_0"] = df1024["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df1024.index.get_level_values('ts')))
df1024["avg90/ts_0"] = df1024["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df1024.index.get_level_values('ts')))
df1024["avg95/ts_0"] = df1024["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df1024.index.get_level_values('ts')))
df1024["avg99/ts_0"] = df1024["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df1024.index.get_level_values('ts')))
df_t1024 = df1024.query(filtered)

# Define parameters
ymin = 2e1
ymax = 4e2
x = np.logspace(-3, -1, 1000)
filtered = '0 < t < 160'
df_t1024 = df1024.query(filtered)
colortxt = 'black'  # Adjust based on your preference

fig = plt.figure(figsize=(3.54331, 3.54331), dpi=300)
gs = gridspec.GridSpec(4, 1, height_ratios=[1.25, 0.2 ,1.25, .75], hspace=0.01)

# First plot
ax1 = fig.add_subplot(gs[0])
ax1.plot(x, np.ones_like(x), color=colortxt, linestyle='dotted')
ax1.vlines(ts_e3_e1_vres, ymin=ymin, ymax=ymax, linestyles='dashed', color=colortxt)
ax1.vlines(ts_e3_e1_avg0, ymin=ymin, ymax=ymax, linestyles='dashdot', color=colortxt)

norm = plt.Normalize(0, 160)
sm = plt.cm.ScalarMappable(cmap='Spectral', norm=norm)
sns.lineplot(data=df_t1024, x="ts", y="avg99/ts", hue='t', ax=ax1, errorbar=('ci', 99.7), hue_norm=norm, palette='Spectral')

ax1.plot(x, sigma_ts0(1e-3, 1e-1, x), color=colortxt, linestyle='dotted')

legend_elements = [Patch(visible=False, label=r'$\mathbf{Analytical}$'),
                   Line2D([0], [0], color=colortxt, linestyle='dashed', label=r'$\tau_s \text{ at } v_{\text{res}}$'),
                   Line2D([0], [0], color=colortxt, linestyle='dashdot', label=r'$\bar{\tau}^0_s$'),
                   Line2D([0], [0], color=colortxt, linestyle='dotted', label=r'$\sigma^0_{(\tau_s)} \propto \tau_s^{-0.5}$')]

ax1.legend(handles=legend_elements, loc='lower left')
ax1.set_xlabel(r'')
ax1.set_ylabel(r'$\bar{\rho}_{(> \!99\%)} \, / \, \tau_\mathrm{s}$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(ymin, ymax)
ax1.set_xlim(1e-3, 1e-1)
plt.setp(ax1.get_xticklabels(), visible=False)

# Second plot
ymin = 1
ymax = 2e1
ax2 = fig.add_subplot(gs[2], sharex=ax1)
ax2.plot(x, np.ones_like(x), color=colortxt, linestyle='dotted')
ax2.vlines(ts_e3_e1_vres, ymin=ymin, ymax=ymax, linestyles='dashed', color=colortxt)
ax2.vlines(ts_e3_e1_avg0, ymin=ymin, ymax=ymax, linestyles='dashdot', color=colortxt)

sns.lineplot(data=df_t1024, x="ts", y="avg99/ts_0", hue='t', ax=ax2, errorbar=('ci', 99.7), hue_norm=norm, palette='Spectral', legend=False)

ax2.set_ylabel('')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(ymin, ymax)
ax2.set_xlim(1e-3, 1e-1)
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.hlines(1.1, xmin=1e-3, xmax=1e-1, linestyles='dashed', color=colortxt, linewidth=.25*lwidth)
ax2.hlines(.95, xmin=1e-3, xmax=1e-1, linestyles='dashed', color=colortxt, linewidth=.25*lwidth)

gs.update(hspace=0)

# Third plot

ymin = .8
ymax = 2e1
ax3 = fig.add_subplot(gs[3], sharex=ax1)
ax3.plot(x, np.ones_like(x), color=colortxt, linestyle='dotted')
ax3.vlines(ts_e3_e1_vres, ymin=ymin, ymax=ymax, linestyles='dashed', color=colortxt)
ax3.vlines(ts_e3_e1_avg0, ymin=ymin, ymax=ymax, linestyles='dashdot', color=colortxt)

sns.lineplot(data=df_t1024, x="ts", y="avg99/ts_0", hue='t', ax=ax3, errorbar=('ci', 99.7), hue_norm=norm, palette='Spectral', legend=False)

ax3.set_xlabel(r'$\tau_\mathrm{s}$')
ax3.set_ylabel(r'')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_yticks([0.98, 0.99, 1, 1.01 ,1.02], [0.98, 0.99, 1, 1.01, 1.02], minor=True)
ax3.tick_params(which='minor', labelsize=5)
ax3.set_ylim(.98, 1.025)
ax3.set_xlim(1e-3, 1e-1)

# Colorbar
cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3], orientation='vertical')
cbar.set_label(r'$\Omega t$')

fig.text(0.02, 0.35, r'$\bar{\sigma}_{(\rho_\mathrm{d} > \!99\%)}/\sigma^0$', ha='center', va='center', rotation='vertical')

plt.savefig('size_time.png')
plt.close()

######################################################
######################################################
######################################################

filtered = '175 < t < 225'
x = np.logspace(-3, -1, 1000)

df_b32= pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_mrn32_dens_ts_cont.csv", index_col=['t', 'ts'])
df_b32["avg68/ts_0"] = df_b32["avg68/ts"]/(sigma_ts0_32(1e-3, 1e-1, df_b32.index.get_level_values('ts')))
df_b32["avg90/ts_0"] = df_b32["avg90/ts"]/(sigma_ts0_32(1e-3, 1e-1, df_b32.index.get_level_values('ts')))
df_b32["avg95/ts_0"] = df_b32["avg95/ts"]/(sigma_ts0_32(1e-3, 1e-1, df_b32.index.get_level_values('ts')))
df_b32["avg99/ts_0"] = df_b32["avg99/ts"]/(sigma_ts0_32(1e-3, 1e-1, df_b32.index.get_level_values('ts')))
df_b32 = df_b32.query(filtered)

df_b35= pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_dens_ts_cont.csv", index_col=['t', 'ts'])
df_b35["avg68/ts_0"] = df_b35["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df_b35.index.get_level_values('ts')))
df_b35["avg90/ts_0"] = df_b35["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df_b35.index.get_level_values('ts')))
df_b35["avg95/ts_0"] = df_b35["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df_b35.index.get_level_values('ts')))
df_b35["avg99/ts_0"] = df_b35["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df_b35.index.get_level_values('ts')))
df_b35 = df_b35.query(filtered)

df_b38= pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_mrn38_dens_ts_cont.csv", index_col=['t', 'ts'])
df_b38["avg68/ts_0"] = df_b38["avg68/ts"]/(sigma_ts0_38(1e-3, 1e-1, df_b38.index.get_level_values('ts')))
df_b38["avg90/ts_0"] = df_b38["avg90/ts"]/(sigma_ts0_38(1e-3, 1e-1, df_b38.index.get_level_values('ts')))
df_b38["avg95/ts_0"] = df_b38["avg95/ts"]/(sigma_ts0_38(1e-3, 1e-1, df_b38.index.get_level_values('ts')))
df_b38["avg99/ts_0"] = df_b38["avg99/ts"]/(sigma_ts0_38(1e-3, 1e-1, df_b38.index.get_level_values('ts')))
df_b38 = df_b38.query(filtered)

ymin = 1e1
ymax = 5e2
fig, ax = plt.subplots()

plt.plot(x, sigma_ts0_38(1e-3, 1e-1, x), color=medium[0], linestyle='dotted')
plt.plot(x, sigma_ts0(1e-3,1e-1, x), color=medium[1], linestyle='dotted')
plt.plot(x, sigma_ts0_32(1e-3, 1e-1, x), color=medium[2], linestyle='dotted')

plt.vlines(ts_e3_e1_vres_mrn32, ymin=ymin, ymax=ymax, linestyles='dashed', color=medium[2])
plt.vlines(ts_e3_e1_vres, ymin=ymin, ymax=ymax, linestyles='dashed', color=medium[1])
plt.vlines(ts_e3_e1_vres_mrn38, ymin=ymin, ymax=ymax, linestyles='dashed', color=medium[0])

sns.lineplot(data=df_b38, x="ts", y="avg99/ts", ax=ax, errorbar=('ci', 99.7), color=medium[0])
sns.lineplot(data=df_b35, x="ts", y="avg99/ts", ax=ax, errorbar=('ci', 99.7), color=medium[1])
sns.lineplot(data=df_b32, x="ts", y="avg99/ts", ax=ax, errorbar=('ci', 99.7), color=medium[2])
           
legend_elments = [Patch(visible=False, label=r'$\mathbf{Slope}$'),
                  Line2D([0], [0], color=medium[0],  label=r'$\sigma^0_{(\tau_s)} \propto \tau_s^{-1/5}$'),
                  Line2D([0], [0], color=medium[1],  label=r'$\sigma^0_{(\tau_s)} \propto \tau_s^{-1/2}$'),
                  Line2D([0], [0], color=medium[2],  label=r'$\sigma^0_{(\tau_s)} \propto \tau_s^{-4/5}$'),
                  Patch(visible=False, label=r'$\mathbf{Analytical}$'),
                  Line2D([0], [0], color=colortxt, linestyle='dashed', label=r'$\tau_s \text{ at } v_{\text{res}}$'),
                  Line2D([0], [0], color=colortxt, linestyle='dotted', label=r'$\sigma^0_{(\tau_s)}$')] 
ax.legend(handles=legend_elments[:], loc='lower left')

ax.set_xlabel(r'$\tau_\mathrm{s}$')
ax.set_ylabel(r'$\bar{\rho}/\tau_s$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(ymin, ymax)
ax.set_xlim(1e-3, 1e-1)

ax.text(.5, 0.92, r'$175 \leq \Omega t \leq 225$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

ax.set_xlabel(r'$\tau_\mathrm{s}$')
ax.set_ylabel(r'$\bar{\rho}_{(> 99\%)}/\tau_s$')
plt.savefig('size_slope.png')
plt.close()

######################################################
######################################################
######################################################

filtered = '150 <= t <= 200 '
x = np.logspace(-3, -1, 1000)

df10 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_dens_ts_cont.csv", index_col=['t','ts'])
df10["avg68/ts_0"] = df10["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df10.index.get_level_values('ts')))
df10["avg90/ts_0"] = df10["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df10.index.get_level_values('ts')))
df10["avg95/ts_0"] = df10["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df10.index.get_level_values('ts')))
df10["avg99/ts_0"] = df10["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df10.index.get_level_values('ts')))
df_t10 = df10.query(filtered)

df10d = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_dens_ts_disc.csv", index_col=['t','ts'])
df10d["avg68/ts_0"] = df10d["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df10d.index.get_level_values('ts')))
df10d["avg90/ts_0"] = df10d["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df10d.index.get_level_values('ts')))
df10d["avg95/ts_0"] = df10d["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df10d.index.get_level_values('ts')))
df10d["avg99/ts_0"] = df10d["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df10d.index.get_level_values('ts')))
df_t10d = df10d.query(filtered)

df20 = pd.read_csv(file_dir+"/processed_data/res256_poly20_ts(1e-3|1e-1)_k30_dens_ts_cont.csv", index_col=['t','ts'])
df20["avg68/ts_0"] = df20["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df20.index.get_level_values('ts')))
df20["avg90/ts_0"] = df20["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df20.index.get_level_values('ts')))
df20["avg95/ts_0"] = df20["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df20.index.get_level_values('ts')))
df20["avg99/ts_0"] = df20["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df20.index.get_level_values('ts')))
df_t20 = df20.query(filtered)

df20d = pd.read_csv(file_dir+"/processed_data/res256_poly20_ts(1e-3|1e-1)_k30_dens_ts_disc.csv", index_col=['t','ts'])
df20d["avg68/ts_0"] = df20d["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, df20d.index.get_level_values('ts')))
df20d["avg90/ts_0"] = df20d["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, df20d.index.get_level_values('ts')))
df20d["avg95/ts_0"] = df20d["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, df20d.index.get_level_values('ts')))
df20d["avg99/ts_0"] = df20d["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, df20d.index.get_level_values('ts')))
df_t20d = df20d.query(filtered)

dfdisc10 = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_disc_dens_ts_cont.csv", index_col=['t','ts'])
dfdisc10["avg68/ts_0"] = dfdisc10["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc10.index.get_level_values('ts')))
dfdisc10["avg90/ts_0"] = dfdisc10["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc10.index.get_level_values('ts')))
dfdisc10["avg95/ts_0"] = dfdisc10["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc10.index.get_level_values('ts')))
dfdisc10["avg99/ts_0"] = dfdisc10["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc10.index.get_level_values('ts')))
df_disc10t = dfdisc10.query(filtered)

dfdisc10d = pd.read_csv(file_dir+"/processed_data/res256_poly10_ts(1e-3|1e-1)_k30_disc_dens_ts_disc.csv", index_col=['t','ts'])
dfdisc10d["avg68/ts_0"] = dfdisc10d["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc10d.index.get_level_values('ts')))
dfdisc10d["avg90/ts_0"] = dfdisc10d["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc10d.index.get_level_values('ts')))
dfdisc10d["avg95/ts_0"] = dfdisc10d["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc10d.index.get_level_values('ts')))
dfdisc10d["avg99/ts_0"] = dfdisc10d["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc10d.index.get_level_values('ts')))
df_disc10td = dfdisc10d.query(filtered)

dfdisc20 = pd.read_csv(file_dir+"/processed_data/res256_poly20_ts(1e-3|1e-1)_k30_disc_dens_ts_cont.csv", index_col=['t','ts'])
dfdisc20["avg68/ts_0"] = dfdisc20["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc20.index.get_level_values('ts')))
dfdisc20["avg90/ts_0"] = dfdisc20["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc20.index.get_level_values('ts')))
dfdisc20["avg95/ts_0"] = dfdisc20["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc20.index.get_level_values('ts')))
dfdisc20["avg99/ts_0"] = dfdisc20["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc20.index.get_level_values('ts')))
df_disc20t = dfdisc20.query(filtered)

dfdisc20d = pd.read_csv(file_dir+"/processed_data/res256_poly20_ts(1e-3|1e-1)_k30_disc_dens_ts_disc.csv", index_col=['t','ts'])
dfdisc20d["avg68/ts_0"] = dfdisc20d["avg68/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc20d.index.get_level_values('ts')))
dfdisc20d["avg90/ts_0"] = dfdisc20d["avg90/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc20d.index.get_level_values('ts')))
dfdisc20d["avg95/ts_0"] = dfdisc20d["avg95/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc20d.index.get_level_values('ts')))
dfdisc20d["avg99/ts_0"] = dfdisc20d["avg99/ts"]/(sigma_ts0(1e-3, 1e-1, dfdisc20d.index.get_level_values('ts')))
df_disc20td = dfdisc20d.query(filtered)


ymin = 1
ymax = 1.2e1
# Create a figure with GridSpec
fig = plt.figure(figsize=(3.54331, 3.54331), dpi=300)
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0)

# First plot
ax1 = fig.add_subplot(gs[0])

ax1.plot(x, np.ones_like(x), color=colortxt, linestyle='dotted')
ax1.vlines(ts_e3_e1_vres, ymin=ymin, ymax=ymax, linestyles='dashed', color=colortxt)

sns.lineplot(data=df_t10, x="ts", y="avg99/ts_0", ax=ax1, errorbar=('ci', 99.7), color=medium[0])
sns.lineplot(data=df_t10d, x="ts", y="avg99/ts_0", ax=ax1, linestyle='', marker='.', markersize=7*lwidth, markeredgewidth=0, errorbar=('ci', 99.7), err_style='bars', color=medium[0])
sns.lineplot(data=df_disc10td, x="ts", y="avg99/ts_0", ax=ax1, linestyle='', marker='.', markersize=7*lwidth, markeredgewidth=0, errorbar=('ci', 99.7), err_style='bars', color=medium[2])
         
legend_elments = [Patch(visible=False, label=r'$\mathbf{n_\mathrm{d}}$'),
                  Line2D([0], [0], marker='.', markersize=10*lwidth, color=medium[0],  label=r'$10$ G.L.'),
                  Line2D([0], [0], marker='o', markersize=10*lwidth, color=colorback, markerfacecolor=medium[2],  label=r'$10$ Discrete'),
                  Patch(visible=False, label=r'$\mathbf{Analytical}$'),
                  Line2D([0], [0], color=colortxt, linestyle='dashed', label=r'$\tau_s \text{ at } v_{\text{res}}$')] 
ax1.legend(handles=legend_elments[:], loc='upper left')

ax1.set_xlabel(r'')
ax1.set_ylabel(r'')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(1e-3,1e-1)
ax1.set_ylim(ymin, ymax)

ax1.text(.5, 0.92, r'$150 \leq \Omega t \leq 200$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

plt.setp(ax1.get_xticklabels(), visible=False)

# First plot
ax2 = fig.add_subplot(gs[1], sharex=ax1)

ax2.plot(x, np.ones_like(x), color=colortxt, linestyle='dotted')
ax2.vlines(ts_e3_e1_vres, ymin=ymin, ymax=ymax, linestyles='dashed', color=colortxt)

sns.lineplot(data=df_t20, x="ts", y="avg99/ts_0", ax=ax2, errorbar=('ci', 99.7), color=medium[1])
sns.lineplot(data=df_t20d, x="ts", y="avg99/ts_0", ax=ax2, linestyle='', marker='.', markersize=7*lwidth, markeredgewidth=0, errorbar=('ci', 99.7), err_style='bars', color=medium[1])

sns.lineplot(data=df_disc20td, x="ts", y="avg99/ts_0", ax=ax2, linestyle='', marker='.', markersize=7*lwidth, markeredgewidth=0, errorbar=('ci', 99.7), err_style='bars', color=medium[4])
         
legend_elments = [Patch(visible=False, label=r'$\mathbf{n_\mathrm{d}}$'),
                  Line2D([0], [0], marker='.', markersize=10*lwidth, color=medium[1],  label=r'$20$ G.L.'),
                  Line2D([0], [0], marker='o',  markersize=10*lwidth, color=colorback, markerfacecolor=medium[4],  label=r'$20$ Discrete'),
                 ] 
ax2.legend(handles=legend_elments[:], loc='upper left')

ax2.set_xlabel(r'')
ax2.set_ylabel(r'')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(1e-3,1e-1)
ax2.set_ylim(ymin, ymax)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'$\tau_\mathrm{s}$')
plt.ylabel(r'$\bar{\sigma}_{(\rho > 99\%)}/\sigma^0$')
plt.savefig('normsize_discrete.png')
plt.close()

######################################################
######################################################
######################################################

n=5
xmin=1e-3
xmax=1e0
def func(x, mu=1e-3, sigma=2.5, xmax=xmax):
    int = spstat.lognorm.cdf(xmax, sigma, loc=mu)
    return spstat.lognorm.pdf(x,sigma, loc=mu)/int
    #return np.ones_like(x)+-0.5*x
x_func = np.logspace(np.log10(xmin),np.log10(xmax),100000)

# Constant spacing in log tau space
xi1 = np.log(xmin)
xi2 = np.log(xmax)

x_log = np.exp(np.linspace(xi1, xi2, n))
x_log_edge = np.linspace(xi1, xi2, n + 1)
x_log = x_log_edge + 0.5*(x_log_edge[1] - x_log_edge[0])
x_log = np.exp(x_log)[0:-1]
log_edge = np.exp(x_log_edge)
int_discrete = np.sum(func(x_log)*np.diff(log_edge))


xi, weights = roots_legendre(n)
xi = np.asarray(xi)
weights = np.asarray(weights)
q = xmax/xmin

# Stopping time nodes
x_gl = xmin*np.power(q, 0.5*(xi + 1))

# Roundabout way to calc 0.5*log(taumax/taumin)
logfac = np.log(q)/(xi[1]-xi[0])

x = xi
notes = 0.5*np.log(q)*weights*x_gl*func(x_gl)
xi = np.linspace(-1, 1, 10000)
res = BarycentricInterpolator(x, func(x_gl))(xi)
logts = np.exp(BarycentricInterpolator(x, np.log(x_gl))(xi))

fig = plt.figure()

plt.plot(x_func,func(x_func), color=colortxt,linestyle='dashed')

plt.stairs(func(x_log), edges=log_edge, baseline=None, color=medium[0])
plt.plot(x_log, func(x_log), '.', color=medium[0])

plt.plot(logts, res, color=medium[1])
plt.plot(x_gl, func(x_gl), '.', color=medium[1])

plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin, xmax)
plt.ylim(1e-1,1e1)
plt.xlabel(r'$x$')
plt.ylabel(r'$f_{(x)}$')

print(sum(notes))

legend_elments = [Patch(visible=False, label=r'Lognormal$(\mu = 10^{-3},\sigma = 2.5)$'),
                  Line2D([0], [0], marker='.', color=medium[0],  label=r'Discrete Method  Integral: {:.5f}'.format(int_discrete)),
                  Line2D([0], [0], marker='.', color=medium[1],  label=r'G.L. Method        Integral: {:.5f}'.format(sum(notes))),
                  Line2D([0], [0], color=colortxt, linestyle='dashed', label=r'Normalized Lognormal')] 
plt.legend(handles=legend_elments[:], loc='lower center')
plt.savefig('example_GL.png')
plt.close()

######################################################
######################################################
######################################################

lwidth=2
cmap = sns.color_palette("flare", as_cmap=True)
figsize2 = (7.48031, 7.48031/1.85)
colorback = 'white'#(32.2/100, 2/100, 48.2/100)
colortxt = 'black'#(88.2/100, 91/100, 92.2/100)
plt.rc('figure', figsize=(7.48031, 7.48031/(16/9)), dpi=300, facecolor=colorback, edgecolor=colorback)
plt.rc('savefig', bbox='tight')
plt.rc('lines', linewidth=lwidth)
plt.rc('xtick', labelsize=7, color=colortxt, top=True) 
plt.rc('xtick.major', size=3.5*lwidth, width=0.8*lwidth)
plt.rc('ytick.major', size=3.5*lwidth, width=0.8*lwidth)
plt.rc('xtick.minor', size=2*lwidth, width=0.6*lwidth)
plt.rc('ytick.minor', size=2*lwidth, width=0.6*lwidth)
plt.rc('ytick', labelsize=7, color=colortxt, right=True)
plt.rc('font', size=12, family='serif', serif='Times New Roman')
plt.rc('mathtext', fontset= 'dejavuserif')

plt.rc('xtick', labelsize=7, color=colortxt, top=True) 
plt.rc('ytick', labelsize=7, color=colortxt, right=True)
plt.rc('axes', grid=False) #grid

fonts = 19
fontm =28
lsize =15


cmap='inferno'
colortxt='black'
vmin =  1e-1
vmax = 1e2
# Sample data
frame = 333
dir = file_dir+'/raw_data/poly10_1024/'
coord = vt.Coordinates(dir)
pf = vt.PolyFluid(dir)
pf0 = vt.PolyFluid(dir)
pf0.read(0)
dt = vt.time_stamps(dir, 2)[1]
X, Z = np.meshgrid(coord.x, coord.z, sparse=False, indexing='xy')


# Create figure with specific dimensions
fig = plt.figure(figsize=(25, 25/1.85), dpi=300)
sig = r'$\bar{\rho}_\mathrm{d}^0$'
sig_g = r'$\bar{\rho}_\mathrm{g}^0$'
tbar = r'$\bar{\tau}_\mathrm{s}^0$'
ts = r'$\tau_\mathrm{s}^0$'

# Define GridSpec with specific spacing adjustments
gs_main = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 3, 0.15], wspace=0.01)

# Define the left gridspec
gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0], hspace=0.05)

# Define the right gridspec
gs_right = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_main[1], wspace=0.12, hspace=0.15)

# Function to configure ticks
def set_ticks(ax, labelleft=False, labelbottom=False):
    ax.tick_params(direction='in', which='both', top=True, bottom=True, left=True, right=True,
                labelleft=labelleft, labelbottom=labelbottom, labelsize=fonts)
    # ax.set_xticks(np.arange(0, 10, 2))
    # ax.set_yticks(np.arange(0, 10, 2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_box_aspect(1)

# Create subplots in the left part
pf0.read(0)
dens0 = pf0.dust_density()[:,0,:]
mean = np.mean(dens0)
pf.read(frame)
dens = pf.dust_density()[:,0,:]/dens0

ax1 = fig.add_subplot(gs_left[0])
heatmap1 = ax1.pcolormesh(coord.x, coord.z, dens, norm=colors.LogNorm(vmin=vmin,vmax=vmax), cmap=cmap)
ax1.text(-.1, 0.125, f'{sig} = {mean:.2f}; {tbar} = {ts_e3_e1_avg0:.2g}', color=colortxt, va='top', ha='left', fontsize=fonts)
set_ticks(ax1, labelleft=True)

pf0.read(0)
pfFluids0 = pf0.Fluids
pf.read(frame)
pfFluids = pf.Fluids

dens0 = pfFluids0[0].dens[:,0,:]
mean = np.mean(dens0)
dens = pfFluids[0].dens[:,0,:]/dens0
ax2 = fig.add_subplot(gs_left[1], sharex=ax1, sharey=ax1)
heatmap2 = ax2.pcolormesh(coord.x, coord.z, dens, norm=colors.LogNorm(vmin=.9999,vmax=1.0001), cmap='viridis')
ax2.text(-.1, -.125, f'{sig_g} = {mean:.2f}', color=colortxt, va='top', ha='left', fontsize=fonts)
# Adding a separate colorbar for ax2
cbar_ax2 = fig.add_axes([0.368, 0.12, 0.015, 0.35])  # Position it next to ax2
cbar2 = fig.colorbar(heatmap2, cax=cbar_ax2, extend='both')
cbar2.ax.tick_params(labelsize=lsize)
cbar2.ax.tick_params(which='minor', size=0, labelsize=0)
cbar2.set_ticks([.9999, 1, 1.0001])
cbar2.set_ticklabels([r'$1-10^{-4}$', r'$1$', r'$1+10^{-4}$'])
fig.text(0.4, 0.3, r'$\rho_\mathrm{g}/\bar{\rho}_\mathrm{g}^{0}$', ha='center', va='center', rotation='vertical',fontsize=20)
set_ticks(ax2, labelleft=True, labelbottom=True)

# Create subplots in the right part
k = 0
for i in range(3):
    for j in range(3):
        if i > 0:
            k +=1
            if j == 0 and i ==1:
                k = -6
            tau = 10 + k
        else:
            tau = k
            k +=1
        
        dens0 = pfFluids0[k].dens[:,0,:]
        mean = np.mean(dens0)
        dens = pfFluids[k].dens[:,0,:]/dens0
        ax = fig.add_subplot(gs_right[i, j])
        heatmap = ax.pcolormesh(coord.x, coord.z, dens, norm=colors.LogNorm(vmin=vmin,vmax=vmax), cmap=cmap)
        ax.text(-.1, .132, f'({tau+1}). {sig} = {mean:.2g}; {ts} = {pf.stopping_times[tau]:.2g}', color=colortxt, va='top', ha='left', fontsize=fonts)

        set_ticks(ax, labelleft=(j==0), labelbottom=(i==2))

# Shared Colorbar for all heatmaps
cbar_ax = fig.add_subplot(gs_main[2])
cbar = fig.colorbar(heatmap, cax=cbar_ax, aspect=100, extend='both')  # aspect makes the colorbar narrower
cbar.ax.tick_params(labelsize=fonts)


# Global axis labels
fig.text(0.55, 0.06, r'$L_x [\pi/K_x]$', ha='center', va='center',fontsize=fontm)
fig.text(0.12, 0.5, r'$L_z [\pi/K_z]$', ha='center', va='center', rotation='vertical', fontsize=fontm)
fig.text(0.93, 0.5, r'$\rho_\mathrm{d}/\bar{\rho}_\mathrm{d}^{0}$', ha='center', va='center', rotation='vertical',fontsize=fontm)

# Adjusted subtitle for the right 9 heatmaps
fig.text(0.65, 0.92, 'Dust species', ha='center', va='center', fontsize=fontm)
fig.text(0.28, 0.93, 'Sum over all dust species', ha='center', va='center', fontsize=fontm)
fig.text(0.28, 0.04, 'Gas', ha='center', va='center', fontsize=fontm)
fig.text(0.55, 0.94, r'$\Omega t = {:.2f}$'.format(dt*(frame)), ha='center', va='center', fontsize=fontm)

plt.savefig('dens0_logfix.png')

######################################################
######################################################
######################################################

frame = 333
lwidth=1
plt.rc('xtick.major', size=3.5*lwidth, width=0.8*lwidth)
plt.rc('ytick.major', size=3.5*lwidth, width=0.8*lwidth)
plt.rc('xtick.minor', size=2*lwidth, width=0.6*lwidth)
plt.rc('ytick.minor', size=2*lwidth, width=0.6*lwidth)
plt.rc('axes', facecolor=colorback, edgecolor=colortxt, titlesize=7, titlecolor=colortxt,  labelsize=7, labelcolor=colortxt, linewidth=.8*lwidth)#fontsize of the title #fontsize of the x and y label

dir = file_dir+'/raw_data/poly10_1024/'
coord = vt.Coordinates(dir)
pf = vt.PolyFluid(dir)
dt = vt.time_stamps(dir, 2)[1]

pf.read(0)
dens_sum0 = np.mean(pf.dust_density()[:,0,:])
dens_high0 = np.mean(pf.Fluids[-1].dens[:,0,:].ravel())
dens_lhigh0 = np.mean(pf.Fluids[-2].dens[:,0,:].ravel())
dens_llhigh0 = np.mean(pf.Fluids[-3].dens[:,0,:].ravel())
dens_low0 = np.mean(pf.Fluids[-4].dens[:,0,:].ravel())

fig, axs = plt.subplots(2,2,figsize=(7.48031,7.48031), dpi=600)

ax1 = axs.ravel()[0]
ax2 = axs.ravel()[1]
ax3 = axs.ravel()[2]
ax4 = axs.ravel()[3]

#upperleft
ax1.set_aspect(1)
ax1.set_xlim(-0.06, -0.06+0.02)
ax1.set_ylim(0.045, 0.045+0.02)
#upperright
ax2.set_aspect(1)
ax2.set_xlim(0.015,0.015+0.02)
ax2.set_ylim(0.0825, 0.0825+0.02)
#lowerleft
ax3.set_aspect(1)
ax3.set_xlim(-0.025, -0.025+0.02)
ax3.set_ylim(-0.0625,-0.0625+0.02)
ax3.set_xticks([-0.025, -0.02,-0.015, -0.01,-0.005])

ax4.set_aspect(1)

pf.read(frame)

##########################a
X, Z = np.meshgrid(coord.x, coord.z, sparse=False, indexing='xy')
X = coord.x
Z = coord.z
PC = 99

dust_dens = pf.dust_density()[:,0,:]
pf.read(frame)
dens_sum = np.percentile(dust_dens,PC)
ax1.contourf(X, Z,dust_dens, [dens_sum, 1e10], colors=[colortxt])
ax2.contourf(X, Z,dust_dens, [dens_sum, 1e10], colors=[colortxt])
ax3.contourf(X, Z,dust_dens, [dens_sum, 1e10], colors=[colortxt])
ax4.contourf(X, Z,dust_dens, [dens_sum, 1e10], colors=[colortxt])

dens_high = np.percentile(pf.Fluids[-1].dens[:,0,:].ravel(), PC)
dh = pf.Fluids[-1].dens[:,0,:]
ax1.contour(X, Z, dh, [dens_high], colors=[medium[0]], alpha=.8)
ax2.contour(X, Z, dh, [dens_high], colors=[medium[0]], alpha=.8)
ax3.contour(X, Z, dh, [dens_high], colors=[medium[0]], alpha=.8)
ax4.contour(X, Z, dh, [dens_high], colors=[medium[0]], alpha=.8)

dens_lhigh = np.percentile(pf.Fluids[-2].dens[:,0,:].ravel(), PC)
dlh  = pf.Fluids[-2].dens[:,0,:]
ax1.contour(X, Z, dlh, [dens_lhigh], colors=[medium[1]], alpha=.7)
ax2.contour(X, Z, dlh, [dens_lhigh], colors=[medium[1]], alpha=.7)
ax3.contour(X, Z, dlh, [dens_lhigh], colors=[medium[1]], alpha=.7)
ax4.contour(X, Z, dlh, [dens_lhigh], colors=[medium[1]], alpha=.7)

dens_llhigh = np.percentile(pf.Fluids[-3].dens[:,0,:].ravel(), PC)
dllh = pf.Fluids[-3].dens[:,0,:]
ax1.contour(X, Z, dllh, [dens_llhigh], colors=[medium[2]], alpha=.7)
ax2.contour(X, Z, dllh, [dens_llhigh], colors=[medium[2]], alpha=.7)
ax3.contour(X, Z, dllh, [dens_llhigh], colors=[medium[2]], alpha=.7)
ax4.contour(X, Z, dllh, [dens_llhigh], colors=[medium[2]], alpha=.7)

dens_low = np.percentile(pf.Fluids[-4].dens[:,0,:].ravel(), PC)
dl =  pf.Fluids[-4].dens[:,0,:]
ax1.contour(X, Z, dl, [dens_low], colors=[medium[4]], alpha=.5)
ax2.contour(X, Z, dl, [dens_low], colors=[medium[4]], alpha=.5)
ax3.contour(X, Z, dl, [dens_low], colors=[medium[4]], alpha=.5)
ax4.contour(X, Z, dl, [dens_low], colors=[medium[4]], alpha=.5)

###################################################################################################################
lowleft = Rectangle((-0.025, -0.0625), 0.02, 0.02, edgecolor=colortxt, facecolor='none', linestyle='dashed', linewidth=lwidth)
ax4.add_patch(lowleft)

ax3.text(0.035, 0.95, 'C',
    horizontalalignment='center',
    verticalalignment='center', fontsize=9,
    transform = ax3.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

ax4.text(0.425, 0.34, 'C',
    horizontalalignment='center',     verticalalignment='center', fontsize=9,
    transform = ax4.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))
###################################################################################################################
upleft = Rectangle((-0.06, 0.045), 0.02, 0.02, edgecolor=colortxt, facecolor='none', linestyle='dashed', linewidth=lwidth)
ax4.add_patch(upleft)

ax1.text(0.035, 0.95, 'A',
    horizontalalignment='center',
    verticalalignment='center', fontsize=9,
    transform = ax1.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

ax4.text(0.16, 0.76, 'A',
    horizontalalignment='center',     verticalalignment='center', fontsize=9,
    transform = ax4.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))
###################################################################################################################
upright = Rectangle((0.015, 0.0825), 0.02, 0.02, edgecolor=colortxt, facecolor='none', linestyle='dashed', linewidth=lwidth)
ax4.add_patch(upright)
ax2.text(0.035, 0.95, 'B',
    horizontalalignment='center',
    verticalalignment='center', fontsize=9,
    transform = ax2.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))

ax4.text(0.52, 0.94, 'B',
    horizontalalignment='center',     verticalalignment='center', fontsize=9,
    transform = ax4.transAxes, bbox=dict(boxstyle="round",  facecolor='white', edgecolor='lightgrey', alpha=.6))
###############################################################

tbar = r'\bar{\tau}_s'
legend_elments = [
                Line2D([0], [0], marker='o', markerfacecolor=colortxt, color=colorback, markersize=12, markeredgewidth=1.5, markeredgecolor=colortxt, label=r'${} = {:.3g}$'.format(tbar,ts_e3_e1_avg0)),
                Line2D([0], [0], marker='o', markerfacecolor=colorback,color=colorback, markersize=12, markeredgewidth=1.5, markeredgecolor=medium[0], alpha=.8, label=r'$\tau_s = {:.2g}$'.format(pf.stopping_times[-1])),
                Line2D([0], [0], marker='o', markerfacecolor=colorback,color=colorback, markersize=12, markeredgewidth=1.5, markeredgecolor=medium[1], alpha=.7, label=r'$\tau_s = {:.2g}$'.format(pf.stopping_times[-2])),
                Line2D([0], [0], marker='o', markerfacecolor=colorback,color=colorback, markersize=12, markeredgewidth=1.5, markeredgecolor=medium[2], alpha=.7, label=r'$\tau_s = {:.2g}$'.format(pf.stopping_times[-3])),
                Line2D([0], [0], marker='o', markerfacecolor=colorback,color=colorback, markersize=12, markeredgewidth=1.5, markeredgecolor=medium[4], alpha=.6, label=r'$\tau_s = {:.2g}$'.format(pf.stopping_times[-4]))]

legend1 = ax4.legend(handles=legend_elments[:], loc='lower left')

legend_elments = [Patch(visible=False, label=r'Contour: $\mathbf{\rho_{\tau_s} > 99\%}$'),
            Line2D([0], [0], marker='o', markerfacecolor=colortxt, color=colorback, markersize=12, markeredgewidth=1.5, markeredgecolor=colortxt,
                    label=r'sum $\rho/\rho^0 > {:.2g}$'.format(dens_sum/dens_sum0)),
            Line2D([0], [0], marker='o', markerfacecolor=colorback,color=colorback, markersize=12, markeredgewidth=1.5, markeredgecolor=medium[0],
                    alpha=.85, label=r'$\rho/\rho^0 > {:.2g}$'.format(dens_high/dens_high0)),
            Line2D([0], [0], marker='o', markerfacecolor=colorback,color=colorback, markersize=12, markeredgewidth=1.5, markeredgecolor=medium[1],
                    alpha=.8, label=r'$\rho/\rho^0 > {:.2g}$'.format(dens_lhigh/dens_lhigh0)),
            Line2D([0], [0], marker='o', markerfacecolor=colorback,color=colorback, markersize=12, markeredgewidth=1.5, markeredgecolor=medium[2],
                    alpha=.75, label=r'$\rho/\rho^0 > {:.2g}$'.format(dens_llhigh/dens_llhigh0)),
            Line2D([0], [0], marker='o', markerfacecolor=colorback,color=colorback, markersize=12, markeredgewidth=1.5, markeredgecolor=medium[4],
                    alpha=.7, label=r'$\rho/\rho^0 > {:.2g}$'.format(dens_low/dens_low0))]

legend2 = ax4.legend(handles=legend_elments[:], loc='lower right')
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'$L_x \ [\pi/K_x]$')
plt.ylabel(r'$L_z \ [\pi/K_z]$')
plt.title(r'$\Omega t$ {:.2f}'.format(dt*(frame)))

plt.savefig('1024_poly10_99PC_{}.png'.format(int(frame)))