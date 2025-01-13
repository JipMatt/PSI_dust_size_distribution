"""
Code: vistools.py
Credit: Sijme-Jan Paardekooper 2023
License: GNU GENERAL Public License (https://www.gnu.org/licenses/)
Source: https://github.com/SijmeJan/psi_fargo/tree/sjp/psi
"""


import numpy as np
from scipy.special import roots_legendre
from scipy.interpolate import BarycentricInterpolator
from os import listdir
from os.path import isdir

class Scalar:
    def __init__(self, filename):
        data = np.loadtxt(filename)
        self.t = data[:,0]
        self.val = data[:,1]

class Coordinates:
    '''
    Reads in the x,z coordinates of the data
    '''
    def __init__(self, direc):
        # Coordinates of cell *edges*
        #x = np.loadtxt(direc + 'domain_x.dat')[3:-4]
        x = np.loadtxt(direc + 'domain_y.dat')[3:-4]
        z = np.loadtxt(direc + 'domain_z.dat')[3:-4]

        dx = 0.0
        if len(x) > 1:
            dx = x[1] - x[0]
        dz = 0.0
        if len(z) > 1:
            dz = z[1] - z[0]

        # Coordinates of cell *centres*
        self.x = x + 0.5*dx
        self.z = z + 0.5*dz

        self.dx = dx
        self.dz = dz

class Fluid:
    '''
    Reads in the data of the monodisperse case
    reading in the density and velosity of the gas in number=0 and dust if number=1 '''
    def __init__(self, direc, number=0):
        coord = Coordinates(direc)

        self.ny = 1
        self.nx = len(coord.x)
        self.nz = len(coord.z)

        self.basename = 'gas'
        if number > 0:
            self.basename = 'dust' + str(number)

        self.basename = direc + self.basename

    def read(self, number):
        fname = self.basename
        number = int(number)
        self.dens = np.fromfile(fname + 'dens' + str(number) + '.dat')
        self.vely = np.fromfile(fname + 'vx' + str(number) + '.dat')
        self.velx = np.fromfile(fname + 'vy' + str(number) + '.dat')
        self.velz = np.fromfile(fname + 'vz' + str(number) + '.dat')

        self.dens = self.dens.reshape(self.nz, self.ny, self.nx)
        self.velx = self.velx.reshape(self.nz, self.ny, self.nx)
        self.vely = self.vely.reshape(self.nz, self.ny, self.nx)
        self.velz = self.velz.reshape(self.nz, self.ny, self.nx)

class PolyFluid:
    '''
    Reads in the data of the monodisperse case
    reading in the density and velosity of the gas in number=0 and dust if number=1 '''
    def __init__(self, direc):
        #print(isdir(direc))
        data = np.genfromtxt(direc + '/variables.par',dtype='str')
        tau = []
        for d in data:
            if d[0].find('INVSTOKES') != -1:
                tau.append(float(d[1]))
        tau = np.sort(1.0/np.asarray(tau))

        self.stopping_times = tau
        self.n_dust = len(tau)

        # Gas fluid
        self.Fluids = [Fluid(direc, number=0)]

        # Dust fluids
        for n in range(1, self.n_dust + 1):
            self.Fluids.append(Fluid(direc, number=n))

    def read(self, number):
        for fluid in self.Fluids:
            fluid.read(number)

    def average_stopping_time(self):
        num = self.Fluids[1].dens*self.stopping_times[0]
        denom = self.Fluids[1].dens

        for n in range(2, self.n_dust + 1):
            num = num + self.Fluids[n].dens*self.stopping_times[n-1]
            denom = denom + self.Fluids[n].dens

        return num/denom

    def dust_density(self):
        num = self.Fluids[1].dens

        for n in range(2, self.n_dust + 1):
            num += self.Fluids[n].dens
        return num
    
    def dust_velx(self):
        num = self.Fluids[1].dens
        velx = self.Fluids[1].dens * self.Fluids[1].velx
        for n in range(2, self.n_dust + 1):
            num +=  self.Fluids[n].dens
            velx += self.Fluids[n].dens * self.Fluids[n].velx
        return velx/num
    
    def dust_vely(self):
        num = self.Fluids[1].dens
        vely = self.Fluids[1].dens * self.Fluids[1].vely
        for n in range(2, self.n_dust + 1):
            num +=  self.Fluids[n].dens
            vely += self.Fluids[n].dens * self.Fluids[n].vely
        return vely/num
    
    def dust_velz(self):
        num = self.Fluids[1].dens
        velz = self.Fluids[1].dens * self.Fluids[1].velz
        for n in range(2, self.n_dust + 1):
            num +=  self.Fluids[n].dens
            velz += self.Fluids[n].dens * self.Fluids[n].velz
        return velz/num

    def size_distribution(self, i, j, k):
        '''
        reconstruct the size distribution
        Parameters
        ----------
        
        '''
        # How to reconstruct the size distribution from nodal values?
        xi, weights = roots_legendre(self.n_dust)
        xi = np.asarray(xi)
        weights = np.asarray(weights)

        # Roundabout way to calc 0.5*log(taumax/taumin)
        logfac = \
          np.log(self.stopping_times[1]/self.stopping_times[0])/(xi[1]-xi[0])

        x = xi
        y = [self.Fluids[1].dens[i,j,k]]

        for n in range(2, self.n_dust +1):
            y.append(self.Fluids[n].dens[i,j,k])

        # Convert to sigma
        y = np.asarray(y)/weights/self.stopping_times/logfac

        xi = np.linspace(-1, 1, 100)
        res = BarycentricInterpolator(x, y)(xi)

        return x, y, xi, res


def time_stamps(direc, length):
    data = np.genfromtxt(direc + '/variables.par',dtype='str')
    dt = 0.0
    n = 1
    for d in data:
        if d[0] == 'DT':
            dt = float(d[1])
        if d[0] == 'NINTERM':
            n = int(d[1])

    return dt*n*np.arange(length)

def max_save(direc):
    n = 0
    for file in listdir(direc):
        if 'gasdens' in file and '_' not in file:
            n += 1

    return n

def Fourier(direc, time_stamps, Kx, Kz):
    '''
    time_stamps: array values of the time
    '''
    coord = Coordinates(direc)

    x, z = np.meshgrid(coord.x, coord.z, sparse=False, indexing='xy')

    xmin = x - 0.5*coord.dx
    zmin = z - 0.5*coord.dz

    expmed = np.exp(-1j*(Kx*x + Kz*z))#r'$\exp{-i(K_x x + K_z z)}$'
    expminx = np.exp(-1j*(Kx*xmin + Kz*z))#r'$\exp{-i(K_x x_- + K_z z)}$'
    expminz = np.exp(-1j*(Kx*x + Kz*zmin))#r$\exp{-i(K_x x + K_z z_-)}$'

    pf = PolyFluid(direc)

    ret = np.empty((len(time_stamps), 4*(pf.n_dust+1)), dtype=np.complex128)

    for t, n in enumerate(time_stamps):
        pf.read(n)

        for indx, fluid in enumerate(pf.Fluids):
            ret[t,4*indx+0] = np.mean(fluid.dens[:,0,:]*expmed)#r'mean$(\rho \exp{-i(K_x x + K_z z)})$'
            ret[t,4*indx+1] = np.mean(fluid.velx[:,0,:]*expminx)#r'mean$(v_x \exp{-i(K_x x_- + K_z z)})$'
            ret[t,4*indx+2] = np.mean(fluid.vely[:,0,:]*expmed)#r'mean$(v_y \exp{-i(K_x x + K_z z)})$'
            ret[t,4*indx+3] = np.mean(fluid.velz[:,0,:]*expminz)#r'mean$(v_z \exp{-i(K_x x + K_z z_-)})$'

    if (Kx == 0 and Kz == 0):
        return ret
    return 2*ret

def log_levels(minlog, maxlog, n_level=100):
    levels = np.logspace(minlog, maxlog, n_level)
    int_level = np.arange(minlog, maxlog)
    cbarticks = np.power(10.0, int_level)
    cbarlabels = [r'$10^{{{:.0f}}}$'.format(x) for x in int_level]
    return levels, cbarticks, cbarlabels
