import os 
import numpy as np


class Grid:
    def __init__(self, NR, NZ, data):

        self.NR = NR 
        self.NZ = NZ

        if not data.shape[0] == NR*NZ*3:
            self.R_e = data[:NR+1]
            self.R_c = data[NR+1:2*NR+1]
            data = data[2*NR+1:]

            self.tan_th_e = data[:NZ+1]
            self.tan_th_c = data[NZ+1:2*NZ+1]
            data = data[2*NZ+1:]

            self.Z = np.outer(self.R_c, self.tan_th_c)
            self.R = np.outer(self.R_c, np.ones_like(self.tan_th_c))
            self.vol = data.reshape(NR, NZ)
        else:
            data = data.reshape(NR, NZ, 3)
            self.R  = data[:,:, 0]
            self.Z  = data[:,:, 1]
            self.vol = data[:,:, 2]

            self.R_c = self.R[:, 0]
            self.tan_th_c = self.Z[0] / self.R_c[0]
            
class WleGrid: 

    def __init__(self, Nwle, Nbands, wle):

        self.Nwle = Nwle
        self.Nbands = Nbands
        self.wle = wle

        n_sub = int((Nwle-1) / Nbands )
        n_rem = int((Nwle-1) - Nbands*n_sub )
        edges = np.zeros(Nbands-1)
        bands = np.zeros(Nbands)
            
        l = 0; 
        for i in range(Nbands-1):
            l += int(n_sub + (i < n_rem)) 
            edges[i] = np.sqrt(wle[l] * wle[l+1]) 
        assert(l + n_sub == Nwle-1) ; 
        for i in range(1, Nbands-1):
            bands[i] = np.sqrt(edges[i-1]*edges[i]) 

        bands[0] = edges[0] * edges[0] / bands[1] 
        bands[Nbands-1] = edges[Nbands-2] * edges[Nbands-2] / bands[Nbands-2] 

        self.bands = bands
        self.band_edges = edges

class FieldData:
    def __init__(self, rho, vR, vphi, vZ):
        self.rho = rho
        self.vR = vR
        self.vphi= vphi
        self.vZ = vZ

    @classmethod
    def create_empty(cls, shape):
        return FieldData(
            np.full(shape, np.nan),
            np.full(shape, np.nan),
            np.full(shape, np.nan),
            np.full(shape, np.nan)
        )
    
    def __getitem__(self, item):
        return FieldData(
            self.rho[item], self.vR[item], self.vphi[item], self.vZ[item]
        )
    
    def __setitem__(self, item, field):
        self.rho[item] = field.rho
        self.vR[item] = field.vR
        self.vphi[item] = field.vphi
        self.vZ[item] = field.vZ

class FieldData1D:
    def __init__(self, Sigma, vR):
        self.Sigma = Sigma
        self.vR = vR

    @classmethod
    def create_empty(cls, shape):
        return FieldData1D(
            np.full(shape, np.nan),
            np.full(shape, np.nan)
        )
    
    def __getitem__(self, item):
        return FieldData1D(
            self.Sigma[item], self.vR[item]
        )
    
    def __setitem__(self, item, field):
        self.Sigma[item] = field.Sigma
        self.vR[item] = field.vR

class TempData:
    def __init__(self, T, J):
        self.T = T
        self.J = J

    @classmethod
    def create_empty(cls, shape, num_bands):
        return TempData(
            np.full(shape, np.nan),
            np.full(tuple(shape) + (num_bands,), np.nan),
        )
    
    def __getitem__(self, item):
        return TempData(
            self.T[item], self.J[item],
        )
    
    def __setitem__(self, item, field):
        self.T[item] = field.T
        self.J[item] = field.J

class SizeGrid:
    def __init__(self, mass, size):
        self.m_e = mass 
        self.a_e = size
        self.m_c = 0.5*(mass[1:] + mass[:-1])

        D = np.log(mass[1:]/mass[:-1])/np.log(size[1:]/size[:-1])
        self.a_c = size[:-1] * (self.m_c/mass[:-1])**(1/D)

class OpacData:
    def __init__(self, wle, sizes, kappa_abs, kappa_sca):
        self.wle = wle
        self.sizes = sizes
        self.kappa_abs = kappa_abs
        self.kappa_sca = kappa_sca

class Molecule:
    def __init__(self, vap, ice):
        self.vap = vap
        self.ice = ice

class CuDiscModel:
    """Read the outputs from a cuDisc simulation
    
    Parameters
    ----------
    sim_dir : string, None
        Base directory of the simulation output.#
    prim_base : sting, default="dens"
        Base string of the files containing the primitive quantities.
        prim_base="dens" is always correct for current code versions.
    """
    def __init__(self, sim_dir=None, prim_base="dens"):
        if sim_dir is None:
            sim_dir = os.getcwd() 
        self.sim_dir = sim_dir

        self.grid = self.load_grid()
        self._prim_base = prim_base

        self._temp_base = 'temp'

    def load_grid(self):
        grid_file = os.path.join(self.sim_dir, '2Dgrid.dat')

        try:
            NR, NZ = np.fromfile(grid_file, dtype=np.intc, count=2)
            data = np.fromfile(grid_file, dtype=np.double, 
                               offset=2*np.dtype(np.intc).itemsize)
        except IOError:
            raise AttributeError("Could not find the grid file (2Dgrid.dat) "
                                 "in the simulation directory")

        return Grid(NR, NZ,data)

    def load_subgrid(self, index):
        grid_file = os.path.join(self.sim_dir, '2Dgrid_sub'+str(index)+'.dat')

        try:
            NR, NZ = np.fromfile(grid_file, dtype=np.intc, count=2)
            data = np.fromfile(grid_file, dtype=np.double, 
                               offset=2*np.dtype(np.intc).itemsize)
        except IOError:
            raise AttributeError("Could not find the grid file (2Dgrid_sub"+str(index)+".dat) "
                                 "in the simulation directory")

        return Grid(NR, NZ,data)
    
    def load_wle_grids(self):
        wle_file = os.path.join(self.sim_dir, 'wle_grid.dat')

        try:
            Nwle, Nbands = np.fromfile(wle_file, dtype=np.intc, count=2)
            wle = np.fromfile(wle_file, dtype=np.double, 
                            offset=2*np.dtype(np.intc).itemsize, count=Nwle)
        except IOError:
            raise AttributeError("Could not find the grid file (2Dgrid.dat) "
                                 "in the simulation directory")
        
        return WleGrid(Nwle, Nbands, wle)

    def load_grain_sizes(self):
        size_file = os.path.join(self.sim_dir, "grains.sizes")
        try:
            m_e, a = np.genfromtxt(size_file, unpack=True)
        except IOError:
            raise AttributeError("Could not find the grain sizes (grains.sizes) "
                                 "in the simulation directory")
        return SizeGrid(m_e, a)
    
    def load_opacity(self):
        opac_file = os.path.join(self.sim_dir, "interp_opacs.dat")

        n_a, n_wle = np.fromfile(opac_file, dtype=np.intc, count=2)
        data = np.fromfile(opac_file, dtype=np.double, 
                           offset=2*np.dtype(np.intc).itemsize)
        data = data.reshape(n_a, n_wle, 2)

        sizes = self.load_grain_sizes().a_c

        return OpacData(None, sizes, data[:,:,0], data[:,:,1])
    
    def load_output_times(self):
        try:
            return np.genfromtxt(os.path.join(self.sim_dir, "2Dtimes.txt"))
        except IOError:
            raise AttributeError("Could not find the output times (2Dtimes.txt) "
                                 "in the simulation directory")

    def load_prims(self, snap_num):

        if self._prim_base is None:
            self._prim_base = self._get_prim_file_base()

        snap_file = os.path.join(self.sim_dir, f'{self._prim_base}_{snap_num}.dat')

        NR, NZ, Ndust = np.fromfile(snap_file, dtype=np.intc, count=3)
        data = np.fromfile(snap_file, dtype=np.double, offset=3*np.dtype(np.intc).itemsize)
            
        # First try the case with surface density:
        try:
            data = data.reshape(NR, 4*(Ndust+1)*NZ + 1)
            Sigma_g = data[:,-1]
            data = data[:,:-1].reshape(NR, NZ, 4*(Ndust+1))
        except ValueError:
            # Read without surface density
            data = data.reshape(NR, NZ, 4*(Ndust+1))
            Sigma_g = None
        
        g = FieldData(data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3])
        if Sigma_g is not None:
            g.Sigma = Sigma_g

        data = data[:,:,4:]
        data = data.reshape(NR, NZ, Ndust, 4)

        d = FieldData(data[:,:,:,0], data[:,:,:,1], data[:,:,:,2], data[:,:,:,3])

        return g, d 

    def load_dens1D(self, snap_num):

        snap_file = os.path.join(self.sim_dir, f'dens1D_{snap_num}.dat')

        NR, Ndust = np.fromfile(snap_file, dtype=np.intc, count=2)
        data = np.fromfile(snap_file, dtype=np.double, offset=2*np.dtype(np.intc).itemsize)

        data = data.reshape(NR, 2*(Ndust+1))

        gas = FieldData1D(data[:,0], data[:,1])

        data = data[:,2:]
        data = data.reshape(NR, Ndust, 2)

        dust = FieldData1D(data[:,:,0],data[:,:,1])

        return gas, dust 

    def load_temp(self, snap_num):
        """Get the temperature and radiation field"""
        temp_file = os.path.join(self.sim_dir, f'{self._temp_base}_{snap_num}.dat')

        NR, NZ, Nbands = np.fromfile(temp_file, dtype=np.intc, count=3)
        data = np.fromfile(temp_file, dtype=np.double, offset=3*np.dtype(np.intc).itemsize)
        data = data.reshape(NR, NZ, Nbands+1)

        return TempData(data[:,:,0], data[:,:,1:])


    def load_all_prim_data(self, subgrid=None):

        if subgrid != None:
            self.grid = self.load_subgrid(subgrid)

        NR, NZ = self.grid.NR, self.grid.NZ
        
        Ndust = self._get_num_dust_species()
        num_snaps = self._get_num_snaps()

        shape = num_snaps, NR, NZ
        gas  = FieldData.create_empty((num_snaps, NR, NZ))
        dust = FieldData.create_empty((num_snaps, NR, NZ, Ndust))

        # Add surface density, if present
        sig_gas = self._check_for_surface_density_data()
        if sig_gas:
            gas.Sigma = np.full((num_snaps, NR), np.nan)

        for n in range(num_snaps):
            try:
                gn, dn = self.load_prims(n)
                gas[n] = gn
                dust[n] = dn

                if sig_gas:
                    gas.Sigma[n] = gn.Sigma

            except IOError:
                pass
        
        if subgrid != None:
            self.grid = self.load_grid()

        return gas, dust

    def load_all_dens1D_data(self, subgrid=None):

        if subgrid != None:
            self.grid = self.load_subgrid(subgrid)

        NR = self.grid.NR
        self._prim_base = 'dens1D'
        
        Ndust = self._get_num_dust_species()
        print(Ndust)
        num_snaps = self._get_num_snaps()

        shape = num_snaps, NR
        gas  = FieldData1D.create_empty((num_snaps, NR))
        dust = FieldData1D.create_empty((num_snaps, NR, Ndust))

        for n in range(num_snaps):
            try:
                gn, dn = self.load_dens1D(n)
                gas[n] = gn
                dust[n] = dn

            except IOError:
                pass

        self._prim_base = 'dens'
        
        if subgrid != None:
            self.grid = self.load_grid()

        return gas, dust
    
    def load_all_temp_data(self, subgrid=None):

        if subgrid != None:
            self.grid = self.load_subgrid(subgrid)

        NR, NZ = self.grid.NR, self.grid.NZ
        
        Nbands = self._get_num_radiation_bands()
        num_snaps = self._get_num_snaps()

        shape = num_snaps, NR, NZ
        temp  = TempData.create_empty((num_snaps, NR, NZ), Nbands)

        for n in range(num_snaps):
            try:
                tn = self.load_temp(n)
                temp[n] = tn

            except IOError:
                pass

        if subgrid != None:
            self.grid = self.load_grid()

        return temp
                
    def load_all_data(self):
        """Load all of the simulation data"""
        gas, dust = self.load_all_prim_data()
        temp = self.load_all_temp_data()

        return self.grid, gas, dust, temp

    def load1Drun(self):
        """Load 1D run"""
        file1D = os.path.join(self.sim_dir, '1Drun.dat')
        data1D = np.genfromtxt(file1D)
        data1D = data1D.reshape(-1, self.grid.R.shape[0], 5)

        Sig_g, Sig_d, v_d, v_g = data1D[:,:,1],data1D[:,:,2],data1D[:,:,3],data1D[:,:,4]

        return Sig_g, Sig_d, v_d, v_g

    def load_mol(self):
        
        num_snaps = self._get_num_snaps()

        file = os.path.join(self.sim_dir, f'mol_0.dat')

        NR, NZ, Ndust = np.fromfile(file, dtype=np.intc, count=3)

        vap = np.zeros((num_snaps, NR, NZ))
        ice = np.zeros((num_snaps, NR, NZ, Ndust,3))
        
        for snap in range(num_snaps):
            file = os.path.join(self.sim_dir, f'mol_{snap}.dat')

            NR, NZ, Ndust = np.fromfile(file, dtype=np.intc, count=3)
            data = np.fromfile(file, dtype=np.double, offset=3*np.dtype(np.intc).itemsize)
            data = data.reshape(NR, NZ, 3*Ndust+1)
            vap[snap] = data[:,:,0]
            ice[snap] = data[:,:,1:].reshape(NR,NZ,Ndust,3)
        
        return Molecule(vap, ice)

    def _get_prim_file_base(self):
        """Work out the file name used for the primitive quants"""
        valid_names = { 'dens', 'prims', 'dens1D'}
        
        files = os.listdir(self.sim_dir)
        for name in valid_names:
            if f'{name}_0.dat' in files:
                return name
        
        raise AttributeError("Could not find valid file containing initial "
                             "primitive quantities. Do you have the wrong "
                             "simulation directory, or a simulation that you "
                             "did not yet run?")
    
    def _get_num_snaps(self):
        base = self._get_prim_file_base()
        
        files = os.listdir(self.sim_dir)
        snap_nums = \
            [ (f[len(base)+1:-4]) for f in files if f.startswith(base+'_') ]
        
        if 'init' in snap_nums:
            snap_nums.remove('init')
            
        if 'restart' in snap_nums:
            snap_nums.remove('restart')

        if len(snap_nums) == 0:
            return 0
        
        snap_nums = [int(f) for f in snap_nums]

        return max(snap_nums) + 1

    def _get_num_dust_species(self):
        if self._prim_base is None:
            self._prim_base = self._get_prim_file_base()

        snaps = [ f for f in os.listdir(self.sim_dir) if f.startswith(self._prim_base+'_') ]
        if len(snaps) == 0:
            return 0        

        snap_file = os.path.join(self.sim_dir, snaps[0])

        if self._prim_base != 'dens1D':
            return np.fromfile(snap_file, dtype=np.intc, count=3)[2]
        else:
            return np.fromfile(snap_file, dtype=np.intc, count=2)[1]
        
    
    def _get_num_radiation_bands(self):
        snaps = [ f for f in os.listdir(self.sim_dir) if f.startswith(self._temp_base+'_') ]
        if len(snaps) == 0:
            return 0     

        snap_file = os.path.join(self.sim_dir, snaps[0])
        return np.fromfile(snap_file, dtype=np.intc, count=3)[2]
    

    def _check_for_surface_density_data(self):
        """Check whether the snapshots contain surface density data"""
        
        snaps = [ f for f in os.listdir(self.sim_dir) if f.startswith(self._prim_base+'_') ]
        
        if len(snaps) == 0:
            return False
        
        snap_file = os.path.join(self.sim_dir, snaps[0])
        NR, NZ, Ndust = np.fromfile(snap_file, dtype=np.intc, count=3)

        data = np.fromfile(snap_file, dtype=np.double, offset=3*np.dtype(np.intc).itemsize)
        try:
            data = data.reshape(NR, 4*(Ndust+1)*NZ + 1)
            return True
        except ValueError:
            return False
        
if __name__ == "__main__":
    disc = CuDiscModel('outputs/run_template')

    g, d = disc.load_all_prim_data()
    temp = disc.load_all_temp_data()

