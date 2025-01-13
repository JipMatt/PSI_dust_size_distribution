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
#SET DIRECTORY: dir_data = '/file/path/raw_data'
dir = "/name/directory/raw_data"
name = "name_procced_data" #naming format: `[res[resolution]_poly[mono/ndust]_ts([stokesrange])_k[wavenumber/WN]_[additinal_parameter]'
monodisperse = False #set true if monodsiperse

#######################################################################################################
#######################################################################################################
#######################################################################################################
import vistools as vt

import numpy as np
import pandas as pd
import scipy as sp
from scipy.special import roots_legendre
from scipy.interpolate import BarycentricInterpolator

def dust_dist_avg(pf, percentile, discrete=False, stokes_range=[1e-3,1e-1]):
    '''
    returns size distribution at the Gaussian Legendere roots and the interpoled points in between legendre roots.
    Parameters
    ----------
    pf: vistools.PolyFluid object
        read in data at a certain time stamp
    percentile: float
        upper perecentile number between 0 and 100
    
    '''
    dust_dens = np.ravel(pf.dust_density()[:,0,:])
    dens = np.percentile(dust_dens, percentile)

    # How to reconstruct the size distribution from nodal values?
    xi, weights = roots_legendre(pf.n_dust)
    xi = np.asarray(xi)
    weights = np.asarray(weights)

    # Roundabout way to calc 0.5*log(taumax/taumin)
    logfac = \
        np.log(pf.stopping_times[1]/pf.stopping_times[0])/(xi[1]-xi[0])

    x = xi
    y = [np.mean(np.ravel(pf.Fluids[1].dens[:,0,:])[dust_dens >= dens])]

    for n in range(2, pf.n_dust +1):
        y.append(np.mean(np.ravel(pf.Fluids[n].dens[:,0,:])[dust_dens >= dens]))

    # Convert to sigma
    if discrete == True:
        # Constant spacing in log tau space
        xi1 = np.log(stokes_range[0])
        xi2 = np.log(stokes_range[1])

        xi_edge = np.linspace(xi1, xi2, pf.n_dust + 1)
        dxi = xi_edge[1] - xi_edge[0]
        x = np.exp(xi_edge + 0.5*dxi)[0:-1]
        y = np.asarray(y)/np.diff(np.exp(xi_edge))
        xi = np.linspace(np.log(stokes_range[0]), np.log(stokes_range[1]), 100)
        res = BarycentricInterpolator(np.log(x), y)(xi)     
    else:
        y = np.asarray(y)/weights/pf.stopping_times/logfac
        xi = np.linspace(-1, 1, 100)
        res = BarycentricInterpolator(x, y)(xi)
    return x, y, xi, res

def fft_time(pf, t, fluid=None):
    '''
    Calculates a fourier amplitude for the largest mode
    Parameters
    ----------
    pf: vistools.PolyFluid object
        read in data at a certain time stamp
    t: array
        number of time step
    fluid: int, default:None
        index fluid
    '''
    amp_fft00 = np.array([])
    amp_fft11 = np.array([])

    for i in range(len(t)):
        pf.read(i)
        if fluid is None:
            density = pf.dust_density()[:,0,:]
        else:
            density = pf.Fluids[fluid].dens[:,0,:]
        dens_fft = sp.fft.fft2(density)
        amp_fft00 = np.append(amp_fft00, dens_fft[0,0])
        amp_fft11 = np.append(amp_fft11, dens_fft[1,1])
    amp_fft = amp_fft11/amp_fft00
    return amp_fft

def find_avg_perc(pf, frames, percentile):
    '''
    Calculates the averega densisity of the array at values above a certain percentile at every timestep
    Parameters
    ----------
    pf: vistools.PolyFluid object
        read in data at a certain time stamp
    frames: int
        number of time steps
    percentile: float
        percentile of gas must be between 0 and 100
    '''
    avg = np.empty((len(pf.Fluids)+1, frames))
    for i in range(frames):
        pf.read(i)
        dust_dens = np.ravel(pf.dust_density()[:,0,:])
        dens = np.percentile(dust_dens, percentile)
        for j in range(len(pf.Fluids)):
            avg[j, i] =  np.mean(np.ravel(pf.Fluids[j].dens[:,0,:])[dust_dens >= dens])
        avg[j+1, i] = np.mean(dust_dens[dust_dens >= dens])
    return avg

def find_maxdens(pf, frames, start1=False):
    '''
    Calculates the maximum densisity of the whole array at every timestep
    Parameters
    ----------
    pf: vistools.PolyFluid object
        read in data at a certain time stamp
    frames: int
        number of time steps
    '''
    maxs = np.empty((len(pf.Fluids)+1, frames))
    if start1 is True:
        for i in range(frames-1):
            pf.read(i+1)
            for j in range(len(pf.Fluids)):
                maxs[j, i] =  np.max(pf.Fluids[j].dens[:,0,:])

            maxs[j+1, i] = np.max(pf.dust_density()[:,0,:])
    else:
        for i in range(frames):
            pf.read(i)
            for j in range(len(pf.Fluids)):
                maxs[j, i] =  np.max(pf.Fluids[j].dens[:,0,:])

            maxs[j+1, i] = np.max(pf.dust_density()[:,0,:])
    return maxs

def size_dis(pf, frames, percentile, discrete=False, stokes_range=[1e-3,1e-1]):
    '''
    Appends the density/ts for all dust species at a given timestep
    Parameters
    ----------
    pf: vistools.PolyFluid object
        read in data at a certain time stamp
    frames: int
        number of timesteps
    percentile: float
        upper perecentile number between 0 and 100
    '''
    t_res = np.zeros_like([])
    t_y = np.zeros_like([])
    for i in range(frames):
        pf.read(i)
        x, y, xi, res = dust_dist_avg(pf, percentile, discrete=discrete, stokes_range=stokes_range)

        t_res = np.append(t_res, res)
        t_y = np.append(t_y, y)

    if discrete==True:
        logts = np.exp(xi)
    else:
        logts = np.exp(BarycentricInterpolator(x, np.log(pf.stopping_times))(xi))   
    return t_res, t_y, logts


###################################################################################################
############################Time series Density and Amplitude######################################
###################################################################################################


def fargo3d_to_csv(dir, name):
    '''
    Creates pandas cvs of the density and amplitude for different stopping_times and density regions, read in from FARGO output of .dat files
    Parameters
    ----------
    dir: str
        directory of saved output FARGO
    name: str
        name of the saved txt files
    mono: boolian, default: False
        statement if the dirctory has multiple dust species

    '''
    frames= vt.max_save(dir)
    pf = vt.PolyFluid(dir)
    t = vt.time_stamps(dir, frames)

    print(f'Number of datapoints for polydisperse with {len(pf.Fluids)-1} dust species is {len(t)} covering {t[-1]} Omega t')

    # #average stoppingtime
    pf.read(0)
    avg_ts0= np.mean(pf.average_stopping_time())
    pf.read(0)

    # amplitude over time
    ampfft = fft_time(pf, t, 0)
    for n, i in enumerate(pf.stopping_times):
        ampfft = np.append(ampfft, fft_time(pf, t, n+1))
    ampfft= np.abs(np.append(ampfft, fft_time(pf, t, None)))


    maxs= find_maxdens(pf, frames)
    maxs = np.ravel(maxs)

    # the avg denstity in upper Percintile over time
    avg_1sigma = find_avg_perc(pf, frames, 68.3)
    avg_2sigma = find_avg_perc(pf, frames, 95.4)
    avg_Pc90 = find_avg_perc(pf, frames, 90)
    avg_Pc99 = find_avg_perc(pf, frames, 99)

    avg_1sigma = np.ravel(avg_1sigma)
    avg_2sigma = np.ravel(avg_2sigma)
    avg_Pc90 = np.ravel(avg_Pc90)
    avg_Pc99 = np.ravel(avg_Pc99)

    # create index
    fluid = np.array(['gas'])
    for i in range(len(pf.stopping_times)):
        fluid = np.append(fluid,f'dust')
    fluid = np.append(fluid, ['sum-dust'])

    stopping_time = np.append(np.append(np.nan, pf.stopping_times), avg_ts0)

    time = np.tile(t, (len(pf.Fluids)+1))
    fluids = np.repeat(fluid, len(t))
    stopping_times = np.repeat(stopping_time, len(t))

    indx  = (np.vstack((time, fluids, stopping_times)))
    index = pd.MultiIndex.from_arrays(indx, names=["t", "fluid", 'ts'])

    data = np.stack((maxs, avg_1sigma, avg_Pc90, avg_2sigma, avg_Pc99, ampfft), axis=-1)
    df = pd.DataFrame(data, index=index, columns=["max", "avg68", "avg90", "avg95", "avg99",  "ampfft"]).sort_index(level=['t', 'fluid', 'ts'])
    
    df.to_csv(f"{name}_dens_amp.csv")
    return None

###################################################################################################
############################histogram_densitydistribution##########################################
###################################################################################################

def fargo3d_to_csv_PDF(dir, name):
    '''
    Creates pandas cvs of the PDF at different densities, read in from FARGO output of .dat files
    Parameters
    ----------
    dir: str
        directory of saved output FARGO
    name: str
        name of the saved txt files
    mono: boolian, default: False
        statement if the dirctory has multiple dust species

    '''
    frames= vt.max_save(dir)
    pf = vt.PolyFluid(dir)
    t = vt.time_stamps(dir, frames)

    print(f'Number of datapoints for polydisperse with {len(pf.Fluids)-1} dust species is {len(t)} covering {t[-1]} Omega t')
    
    # #average stoppingtime
    pf.read(0)
    avg_ts0= np.mean(pf.average_stopping_time())

    nbins = 100
    max_dens = 1e2
    min_dens = 1e-3
    bins = np.linspace(np.log(min_dens), np.log(max_dens), nbins)
    npixels = len(np.ravel(pf.Fluids[0].dens[:,0,:]))
    dens0 = np.empty((len(pf.Fluids)+1, npixels))
    pdf = np.empty(0)
    pdf0 = np.empty(0)
    cdf = np.empty(0)
    cdf0 = np.empty(0)

    for i in range(len(t)):
        pf.read(i)
        for j in range(len(pf.Fluids)):
            if i ==0:
                dens0[j, :] = np.log(np.ravel(pf.Fluids[j].dens[:,0,:]))
            dens = np.log(np.ravel(pf.Fluids[j].dens[:,0,:]))
            pdf_j, bin_edges = np.histogram(dens, bins=bins, density=True)
            cdf_j = np.cumsum(pdf_j*(np.exp(bins[1:])-np.exp(bins[:-1])))

            pdf0_j, bin_edges = np.histogram(dens-dens0[j,:], bins=bins, density=True)
            cdf0_j = np.cumsum(pdf0_j*(np.exp(bins[1:])-np.exp(bins[:-1])))
            
            pdf = np.append(pdf, pdf_j)
            pdf0 = np.append(pdf0, pdf0_j)
            cdf = np.append(cdf, cdf_j)
            cdf0 = np.append(cdf0, cdf0_j)
        if i ==0:
            dens0[j+1, :] = np.log(np.ravel(pf.dust_density()[:,0,:]))
        dens = np.log(np.ravel(pf.dust_density()[:,0,:]))
        pdf_j, bin_edges = np.histogram(dens, bins=bins, density=True)
        cdf_j = np.cumsum(pdf_j*(np.exp(bins[1:])-np.exp(bins[:-1])))
        pdf0_j, bin_edges = np.histogram(dens-dens0[j+1,:], bins=bins, density=True)
        cdf0_j = np.cumsum(pdf0_j*(np.exp(bins[1:])-np.exp(bins[:-1])))

        pdf = np.append(pdf, pdf_j)
        pdf0 = np.append(pdf0, pdf0_j)
        cdf = np.append(cdf, cdf_j)
        cdf0 = np.append(cdf0, cdf0_j)


    # create index
    fluid = np.array(['gas'])
    for i in range(len(pf.stopping_times)):
        fluid = np.append(fluid,f'dust')
    fluid = np.append(fluid, ['sum-dust'])

    stopping_time = np.append(np.append(np.nan, pf.stopping_times), avg_ts0)

    time = np.repeat(t,  (len(pf.Fluids)+1)*(nbins-1))
    l_edge = np.exp(np.tile(bin_edges[:-1], (len(pf.Fluids)+1)*len(t)))
    u_edge = np.exp(np.tile(bin_edges[1:], (len(pf.Fluids)+1)*len(t)))
    cent =(l_edge+u_edge)/2

    fluids = np.tile(np.repeat(fluid, (nbins-1)), len(t))
    stopping_times = np.tile(np.repeat(stopping_time, (nbins-1)), len(t))

    data = np.stack((np.ravel(pdf), np.ravel(pdf0), np.ravel(cdf), np.ravel(cdf0)), axis=-1)

    indx  = np.vstack((time, fluids, stopping_times, l_edge, cent, u_edge))
    index = pd.MultiIndex.from_arrays(indx, names=["t", "fluid", 'ts', "lbin", "cbin", "ubin"])

    df2 = pd.DataFrame(data, index=index, columns=["pdf", "pdf0",'cdf', 'cdf0']).sort_index(level=['t', "cbin", "lbin", "ubin", 'fluid', 'ts'])
    df2.to_csv(f"{name}_hist.csv")
    return None

###################################################################################################
#######################################density/stoppingtime########################################
###################################################################################################

def fargo3d_to_csv_dens_ts(dir, name, discrete=False):
    '''
    Creates pandas cvs of the density and amplitude for different stopping_times and density regions, read in from FARGO output of .dat files
    Parameters
    ----------
    dir: str
        directory of saved output FARGO
    name: str
        name of the saved txt files
    mono: boolian, default: False
        statement if the dirctory has multiple dust species

    '''
    frames= vt.max_save(dir)
    pf = vt.PolyFluid(dir)
    t = vt.time_stamps(dir, frames)
    print(f'Number of datapoints for polydisperse with {len(pf.Fluids)-1} dust species is {len(t)} covering {t[-1]} Omega t')

    res_pc68, y_pc68, logts = size_dis(pf, frames, 68, discrete=discrete)
    res_pc90, y_pc90, logts = size_dis(pf, frames, 90, discrete=discrete)
    res_pc95, y_pc95, logts = size_dis(pf, frames, 95, discrete=discrete)
    res_pc99, y_pc99, logts = size_dis(pf, frames, 99, discrete=discrete)

    time_y = np.repeat(t, len(pf.stopping_times))
    ts = np.tile(pf.stopping_times, len(t))
    time_cont = np.repeat(t, len(logts))
    ts_cont = np.tile(logts, len(t))

    indx_y  = np.vstack((time_y, ts))
    index_y = pd.MultiIndex.from_arrays(indx_y, names=["t", "ts"])
    indx_cont  = np.vstack((time_cont, ts_cont))
    index_cont = pd.MultiIndex.from_arrays(indx_cont, names=["t", "ts"])

    data_ts = np.stack((y_pc68, y_pc90, y_pc95, y_pc99), axis=-1)
    data_cont = np.stack((res_pc68, res_pc90, res_pc95, res_pc99), axis=-1)

    df_disc = pd.DataFrame(data_ts, index=index_y, columns=["avg68/ts", "avg90/ts", "avg95/ts", "avg99/ts"]).sort_index(level=['t',"ts"])
    df_cont = pd.DataFrame(data_cont, index=index_cont, columns=["avg68/ts", "avg90/ts", "avg95/ts", "avg99/ts"]).sort_index(level=['t',"ts"])

    df_disc.to_csv(f"{name}_dens_ts_disc.csv")
    df_cont.to_csv(f"{name}_dens_ts_cont.csv")

    return None


###################################################################################################
###############################Running the tree functions##########################################
###################################################################################################

fargo3d_to_csv(dir, name)
fargo3d_to_csv_PDF(dir, name)
if not monodisperse:
    fargo3d_to_csv_dens_ts(dir, name)#commend out if proccessing a mondispers

