"""
AERO2564 Tools
2021
Authors: N. Mason-Smith & Q. Michalski
This module includes functions relevant for solving thermofluid problems using Cantera in AERO2564

List of functions:
    HS - compute the pressure corresponding to a thermodynamic state defined by enthalpy-entropy inputs (with composition from a Cantera gas object). Gas object state unchanged. 
    _HSDiff - a helper function for HS
    frozenSoundSpeed -  compute the frozen (chemically) sound speed for a Cantera gas object 
    reversibleChokedNozzle   -  determine the critical pressure of a Cantera gas object (assumed to be at stagnation conditions) through an adiabatic, reversible nozzle. Gas object state unchanged.
    _diffChoking - a helper function for reversibleChokedNozzle and irreversibleChokedNozzle
    irreversibleChokedNozzle - determine the critical pressure of a Cantera gas object (assumed to be at stagnation conditions) through an adiabatic, irreversible nozzle. Gas object state unchanged.
    determineFuelOxidiserRatio - determine the fuel-oxidiser ratio for a gas object (assumed to comprise the oxidiser) and a fuel (assumed to be at the same temperature and pressure as the oxidiser) that gives a particular flame temperature.
    _diffTemperature - a helper function for fuelOxidiserRatio

Test functions also included:
    testHS
    testReversibleChokedNozzle
    testIrreversibleChokedNozzle
    testDetermineFuelOxidiserRatio
"""

import math
from scipy.optimize import root
import cantera as ct

__version__ = '1.0'

### Topic 4 onwards - thermodynamics ###

### Enthalpy-entropy inputs for ideal gas mixtures in Cantera ###
def HS(gas,H,S,initialGuess=1.001):
    # Store the gas' initial state
    h1 = gas.enthalpy_mass
    P1 = gas.P
    # Update gas to the provided enthalpy, at constant pressure (we are going to rootfind for the pressure that gives us the entropy we handed as an argument)
    gas.HP = H,gas.P
    # Now rootfind for the pressure that satisfies the H-S inputs provided - note that HSDiff returns the gas to this state (H , P1)
    sol = root(_HSDiff,[gas.P*initialGuess],args=(gas,S))
    # The solution returned by the rootfinder is the pressure satisfying these inputs
    P = sol["x"][0]
    # Return the gas to its initial state
    gas.HP = h1,P1
    return P

def _HSDiff(Pguess,gas,S):
    # Store gas current state
    P1 = gas.P
    h1 = gas.enthalpy_mass
    # Update to guessed pressure at constant enthalpy
    gas.HP = h1,Pguess
    # Determine the difference between specified S and s(h,Pguess)
    error = gas.entropy_mass-S
    # Return gas to initial state
    gas.HP = h1,P1
    return error

### Frozen sound speed ###
### Adapted from Cantera example code ### 
def frozenSoundSpeed(gas):
    import math
    """
    Returns the frozen sound speed for a gas by using a finite-difference approximation of the derivative.
    """
    # Save properties
    s0 = gas.s
    p0 = gas.P
    r0 = gas.density
    # perturb the pressure
    p1 = p0*1.0001
    # set the gas to a state with the same entropy and composition but
    # the perturbed pressure
    gas.SP = s0, p1
    # frozen sound speed
    afrozen = math.sqrt((p1 - p0)/(gas.density - r0))
    # Return the gas object to its original state 
    gas.SP = s0,p0
    return afrozen


### Topic 5 onwards - compressible flow ###

### Adiabatic reversible choked nozzles ###
# Returns the throat pressure (this can be used in a SP update to find other properties) ##
def reversibleChokedNozzle(gas):
    # Save properties at initial gas state
    s0,P0 = gas.SP
    # Stagnation enthalpy
    stagnationEnthalpy = gas.enthalpy_mass
    # Solve for the throat pressure at which the velocity (from adiabatic reversible expansion) equals the sound speed
    sol = root(_diffChoking,[gas.P*0.9],args=gas)
    throatPressure=sol['x'][0]
    # Update the gas object back to to the stagnation conditions
    gas.SP = s0,P0
    return throatPressure

def _diffChoking(throatPressure,gas,isentropicEfficiency=1):
    """
    Returns the difference between the sound speed of a gas and the velocity predicted by the adiabatic one-dimensional energy equation.
    Can be called by a root-finding algorithm to drive this difference to zero (and hence find the conditions at which the velocity is equal to the speed of sound)
    """
    # Evaluate two independent intensive properties of the gas (so we can reset it to its initial state before exiting the function)
    initialEntropy,stagnationPressure = gas.SP
    # Evaluate the stagnation enthalpy of the gas
    stagnationEnthalpy = gas.enthalpy_mass
    # Update the gas to the estimated throat pressure (isentropic expansion to throatPressure)
    gas.SP = gas.entropy_mass,throatPressure
    # From the definition of the isentropic efficiency:
    h2s = gas.enthalpy_mass
    # Throat enthalpy is higher in the irreversible expansion to the same final pressure
    h2 = stagnationEnthalpy - isentropicEfficiency*(stagnationEnthalpy-h2s)
    # Calculate the specific kinetic energy achieved during this expansion
    specificKineticEnergy = stagnationEnthalpy-h2
    # Velocity from specific kinetic energy
    velocity = math.sqrt(2*specificKineticEnergy)
    # Find the sound speed of the gas at this condition
    soundSpeed = frozenSoundSpeed(gas)
    # Return the gas to its initial conditions
    gas.SP = initialEntropy,stagnationPressure
    # The error term for choking is the difference between the velocity at the end of the expansion and the sound speed at the end of the expansion
    error = soundSpeed-velocity
    return error

### Adiabatic reversible nozzle (checking for choking) ###
def irreversibleChokedNozzle(gas,nozzleIsentropicEfficiency):
    # Save properties
    s0,P0 = gas.SP
    # Stagnation enthalpy
    stagnationEnthalpy = gas.enthalpy_mass
    # Find the throat pressure which results in choked flow for this irreversible nozzle
    sol = root(_diffChoking,[gas.P*0.9],args=(gas,nozzleIsentropicEfficiency))
    throatPressure=sol['x'][0]
    # Update the gas object to the stagnation conditions
    gas.SP = s0,P0
    return throatPressure


### Topic 6 onwards - Combustion ###
### Equivalence ratio that gives you a particular flame temperature ###
def determineFuelOxidiserRatio(gas,fuel,flameTemperature,initialGuess=0.001):
    '''
    A function to determine the fuel-air ratio that gives a target flame temperature, using a root-finding algorithm
    Inputs:
    -gas - a Cantera gas object representing the gas to which fuel is to be added
    -fuel - a string representing the fuel (must be present in the species list of the gas object)
    -flameTemperature - the target temperature at which to find the fuel-oxidiser ratio
    -initialGuess - the initial guess for the root-finding algorithm. If unspecified, defaults to 0.001 (finding the lean root); can be passed as a keyword argument. Depending on the value we may find the rich root or the lean root (or no root, if the reactants aren't sufficiently energetic)
    Outputs:
    -fuelOxidiserRatio - the fuel-air ratio that gives the target flameTemperature
    '''
    # Use a root-finding algorithm to determine the fuel-oxidiser ratio which gives the target flame temperature
    sol = root(_diffTemperature,[initialGuess],args=(gas,fuel,flameTemperature))
    fuelOxidiserRatio = sol['x'][0]
    return fuelOxidiserRatio
    
def _diffTemperature(fuelOxidiserRatio,gas,fuel,flameTemperature):
    ''' 
    A function to return the difference between a target turbine inlet temperature and the adiabatic flame temperature of a fuel-air mixture at a given fuel-air ratio
    Inputs:
    -fuelAirRatio - the mass-based fuel-air ratio (dimensionless)
    -gas - a Cantera gas object containing the gas to which fuel is to be added
    -turbineInletTemperature - the target turbine inlet temperature (K)
    Outputs:
    -error - the difference between the adiabatic flame temperature of gas+fuel (at fuelAirRatio) and turbineInletTemperature (K)'''
    # Store the initial state
    T = gas.T
    P = gas.P
    Y = gas.Y
    # Add fuel to the composition of the gas
    Y_dictionary = gas.mass_fraction_dict()
    Y_dictionary[fuel] = fuelOxidiserRatio
    # Note that this method assumes the fuel is at the same temperature as the incoming air
    gas.TPY = T,P,Y_dictionary
    gas.equilibrate('HP')
    # The error is the difference between the adiabatic flame temperature and the turbine inlet temperature
    error = gas.T-flameTemperature
    # Restore the gas to its initial state
    gas.TPY = T,P,Y
    return error






### Test functions ###
def testHS():
    print('Test function for HS \n Uses air at 101325 Pa, 298.15 K and Mach 1.5, use enthalpy-entropy inputs to determine stagnation pressure')
    # Find the stagnation pressure of a Mach one airflow
    gas = ct.Solution('airNASA9.cti')
    gas.TPX = 298.15,101325,'O2:1 N2:3.76'
    machNumber = 1.5
    # Get velocity from frozenSoundSoeed
    velocity = frozenSoundSpeed(gas) * machNumber
    stagnationEnthalpy = gas.enthalpy_mass + velocity**2/2
    # Obtain stagnation pressure using HS
    stagnationPressure = HS(gas,stagnationEnthalpy,gas.entropy_mass)
    # Determine the ratio to the static pressure
    stagnationPressureRatio = stagnationPressure / gas.P
    # Now compare with perfect gas stagnation pressure ratio for Mach one
    gamma = 1.4
    perfectGasStagnationPressureRatio = (1+(gamma-1)/2*machNumber**2)**(gamma/(gamma-1))
    print('Stagnation pressure ratio from HS function = {} \n Stagnation pressure ratio for perfect gas air = {}'.format(stagnationPressureRatio,perfectGasStagnationPressureRatio))
    return 

def testReversibleChokedNozzle():
    print('Test function for reversible choked nozzle \n Uses air at 101325 Pa, 298.15 K and determines throat pressure (and critical pressure ratio)')
    gas = ct.Solution('airNASA9.cti')
    gas.TPX = 298.15,101325,'O2:1 N2:3.76'
    reservoirPressure = gas.P
    throatPressure = reversibleChokedNozzle(gas)
    criticalPressureRatio = throatPressure/reservoirPressure
    # Perfect gas critical pressure ratio
    gamma = 1.4
    perfectGasCriticalPressureRatio = (1+(gamma-1)/2)**-(gamma/(gamma-1))
    print('Critical pressure ratio from reversibleChokedNozzle = {} \n Critical pressure ratio for perfect gas air = {}'.format(criticalPressureRatio,perfectGasCriticalPressureRatio))
    return

def testIrreversibleChokedNozzle():
    print('Test function for irreversible choked nozzle \n Uses air at 101325 Pa, 298.15 K, nozzle isentropic efficiency 0.5 and determines throat pressure (and critical pressure ratio)')
    gas = ct.Solution('airNASA9.cti')
    nozzleIsentropicEfficiency = 0.5
    gas.TPX = 298.15,101325,'O2:1 N2:3.76'
    reservoirPressure = gas.P
    throatPressure = irreversibleChokedNozzle(gas,0.5)
    criticalPressureRatio = throatPressure/reservoirPressure
    # Perfect gas critical pressure ratio for irreversible nozzle - with a lot of algebra, this is (1 - 
    gamma = 1.4
    perfectGasCriticalPressureRatio = (1-(1-(1+(gamma-1)/2))/nozzleIsentropicEfficiency)**-(gamma/(gamma-1))
    print('Critical pressure ratio from irreversibleChokedNozzle = {} \n Critical pressure ratio for perfect gas air = {}'.format(criticalPressureRatio,perfectGasCriticalPressureRatio))
    return

def testDetermineFuelOxidiserRatio():
    print('Test function for fuelOxidiserRatio \n Uses air at 101325 Pa, 298.15 K, and determines the fuel-air ratio (by mass) with methane which gives a 1601 K flame (doing the inverse problem, using a fuel-air ratio of 0.033 gives a 1601 K flame)') 
    # Find the fuel-air ratio of a methane flame at 1600 K
    gas = ct.Solution('gri30.cti')
    gas.TPX = 298.15,101325,'O2:1 N2:3.76'
    fuelOxidiserRatio = determineFuelOxidiserRatio(gas,'CH4',1601)
    print('Determined fuel/air ratio = {} \n From inverse problem, fuel-air ratio was 0.033'.format(fuelOxidiserRatio))
    return 

def testDetermineFuelOxidiserRatioAlt():
    print('Test function for fuelOxidiserRatio \n Uses air at 101325 Pa, 298.15 K, and determines the fuel-air ratio (by mass) with methane which gives a 1601 K flame (doing the inverse problem, using a fuel-air ratio of 0.033 gives a 1601 K flame)') 
    # Find the fuel-air ratio of a methane flame at 1600 K
    import numpy as np
    gas = ct.Solution('gri30.cti')
    gas.TPX = 298.15,101325,'O2:1 N2:3.76'
    # Iterate over different initial guesses
    initialGuesses = np.linspace(0.001,0.25,125)
    fuelOxidiserRatio = np.empty((len(initialGuesses),))
    for i,initialGuess in enumerate(initialGuesses):
        fuelOxidiserRatio[i] = determineFuelOxidiserRatio(gas,'CH4',1601,initialGuess = initialGuess)
        # print('Determined fuel/air ratio = {} \n From inverse problem, fuel-air ratio was 0.033'.format(fuelOxidiserRatio))
    from matplotlib import pyplot as plt
    plt.scatter(initialGuesses,fuelOxidiserRatio)
    plt.show()
    return 
