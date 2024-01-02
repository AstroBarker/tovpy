'''
Equation of state routines.

The approach here is to construct a class for our equation of state 
that holds necessary information (K & gamma, or the table). 
Then Necessary functions are included to compute necessary quantities.
The API for all functions is the same between polytropic and tabulated 
forms.
'''

import h5py
import numpy as np
from scipy.optimize import brentq

from Interpolator import LogInterpolate_Linear, Interpolate_Linear, Index1D

def Initialize_EOS( eos, *args ):
    """
    This is called to initialize the needed EoS class.

    Parameters:
    -----------
    eos : string - "Polytropic" or "TABLE"
    *args :  list of necessary data
        Polytropic: args = [K, gamma]
        TABLE: nothing
    """

    tables = [ "ls220", "sfho", "sfhx", "dd2", "hshenfsu", "hshen" ]

    if( eos.lower() == "polytropic" ):
        K = args[0]
        gamma = args[1]
        
        eos_object = EoS_Polytropic( eos, K, gamma )
    elif( eos.lower() in tables ):
        eos_object = EoS_TABLE( eos )
    else:
        raise ValueError("Please use polytropic or table EoS")
    return eos_object


# === POLYTROPIC EOS === #

class EoS_Polytropic:
    """
    Hold some EoS constants such as polytropic constant, adiabatic index
    """

    def __init__(self, eos, K, gamma ):
        self.eos   = eos # string - "Polytropic" or "TABLE"
        self.K     = K
        self.gamma = gamma

    def ComputePressure( self, rho ):
        """
        Basic Polytropic pressure

        Parameters:
        ----------
        K     : Polytropic constant ( use 30000 in geometric units)
        rho   : Density
        gamma : Adiabatic index (use 2.75)
        """

        return self.K * rho**(self.gamma)


    def ComputeSpecificInternalEnergy( self, P, rho ):
        """
        Compute the specific internal snergy assuming polytropic EoS

        Parameters:
        -----------
        P : pressure
        rho : density
        """

        tmp = (self.gamma - 1.0) * rho
        return P / tmp


    def ComputeDensity( self, P ):
        """
        Invert polytropic EoS to compute density from pressure

        Parameters:
        ----------
        P     : Pressure
        """

        exp = 1.0 / self.gamma
        return (P / self.K)**exp


# === TABULATED EOS === #

class EoS_TABLE:
    """
    EoS class for a tabulated EoS from StellarCollapse.

    Contains:
    ---------
    Initialization reads and table and rho, T, ye arrays.


    """

    def __init__( self, eos ):
        """
        Reads in an EoS table.
        """

        self.eos = eos
        fn = "../Tables/" + eos + '.h5'

        self.ReadTable( fn )
        self.BetaTable()

    def ReadTable( self, fn ):


        with h5py.File(fn, 'r') as f:
            self.ye           = f["ye"][:]
            self.mu_e         = np.transpose(f["mu_e"][:,:,:])
            self.mu_n         = np.transpose(f["mu_n"][:,:,:])
            self.mu_p         = np.transpose(f["mu_p"][:,:,:])
            self.mu_nu        = np.transpose(f["munu"][:,:,:])
            self.log_rho      = f["logrho"][:]
            self.log_temp     = f["logtemp"][:]
            self.log_energy   = np.transpose(f["logenergy"][:,:,:])
            self.log_pressure = np.transpose(f["logpress"][:,:,:])

            self.energy_offset   = f["energy_shift"][0]
            self.pressure_offset = 0.0
            self.mu_offset       = 0.0


    def BetaTable( self ):
        """
        This gives us an equilibrium Pressure(rho) table.
        This is used in ComputeDensity
        """

        self.P_eq = np.zeros( len(self.log_rho) )

        # Now, we need balance between mu_e and mu_hat
        n_rho = len( self.log_rho )
        for i in range( n_rho ):
            self.P_eq[i] = self.ComputePressure( 10**self.log_rho[i] )


    def BetaEquilibrium( self, Ye, rho ):
        """
        Given a density and ye, return interpolated mu_nu(rho, ye)
        In beta equilibrium, this is zero.
        """

        my_mu_nu = Interpolate_Linear( rho, 10**self.log_temp[0], Ye, \
            self.log_rho, self.log_temp, self.ye, \
            self.mu_nu, self.mu_offset )
        
        return my_mu_nu
    
    def FindYe( self, rho, minYe, maxYe ):
        if( self.BetaEquilibrium( minYe, rho ) * self.BetaEquilibrium(maxYe, rho) > 0.0 ):
            return 2.0*minYe
        else:
            return brentq( self.BetaEquilibrium, minYe, maxYe, args=(rho) )
        

    # === EoS Calls ===

    def ComputePressure( self, rho ):
        """
        Assume T = Tmin, Ye = Ye_min, and interpolate pressure given rho
        """
        
        T = 10**self.log_temp[0]
        Y =     self.FindYe(rho, self.ye[0], self.ye[-1])

        P = LogInterpolate_Linear( rho, T, Y, \
            self.log_rho, self.log_temp, self.ye, \
            self.log_pressure, self.pressure_offset )
        
        return P   


    def ComputeSpecificInternalEnergy( self, P, rho ):
        """
        Computes specific internal energy from the table.
        Pressure is taken as an argument but is not used here, 
        to match syntax of the Polytropic
        """

        T = 10**self.log_temp[0]
        Y =     self.FindYe(rho, self.ye[0], self.ye[-1])

        Em = LogInterpolate_Linear( rho, T, Y, \
             self.log_rho, self.log_temp, self.ye, \
             self.log_energy, self.energy_offset )

        return Em


    def ComputeDensity( self, P ):
        """
        Uses Brent's method to find a new density consistent with 
        our updated pressure. 
        """

        def func( guess ):
            """
            Rooting finding function. 
            Returns zero when we have the correct pressure
            """
            current = self.ComputePressure( 10**guess )
            return P - current

        # Locate some pressures around our current Pressure
        Ps = self.P_eq
        iP = Index1D( P, Ps )

        # Use those pressure indices to get our bounding interval
        if( iP != 0):
            a = self.log_rho[iP]
            b = self.log_rho[iP+1]
        # If we are at the "bottom" of the table, return the min density
        else:
            return 10**self.log_rho[0]

        log_rho = brentq( func, a, b )
        
        return 10**log_rho