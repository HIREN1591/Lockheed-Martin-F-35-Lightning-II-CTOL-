# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:51:14 2023

@author: HIREN GOHIL
"""

### Pratt & Whitney F135-PW-100 Engine 

# Importing libraries
from fluids import atmosphere as atm
import numpy as np
import cantera as ct
import aero2564tools 
from aero2564tools import HS,frozenSoundSpeed,determineFuelOxidiserRatio
from scipy.optimize import root
import matplotlib.pyplot as plt

# Twin spool turbofan engine
# Given data
altitude = int(input("Enter altitude between 0 to 40000 ft (in feet): "))

# Determine aircraft speed based on altitude
if 0 <= altitude < 5000:
    Ma = 0.30   # mach number
elif 5000 <= altitude < 24000:
    Ma = 0.50  # mach number
elif 24000 <= altitude < 40000:
    Ma = 0.85   # mach number 
elif 40000 <= altitude <= 50000:
    Ma = 0.86   # mach number 
else:
    print("Invalid altitude entered.")
    exit()

# Print the determined aircraft speed
if Ma:
    print(f"Aircraft speed at {altitude} feet is {Ma} mach.")

# Create a JetA species
JetA = ct.Species('Jet-A(g)','C:12, H:23')
JetA.thermo = ct.NasaPoly2(273.15,5000,101325,
                                  (1000, # defining the midpoint of the thermodynamic data
                                   2.086921700E+00,   1.331496500E-01, 
                                   -8.115745200E-05,   2.940928600E-08,  -6.519521300E-12,
                                   -3.591281400E+04,   2.735529720E+01, # Up to here are coefficients from 273.15 to 1000 K
                                   2.488020100E+01,   7.825004800E-02, 
                                   -3.155097300E-05,   5.787890000E-09,  -3.982796800E-13,
                                   -4.311068400E+04,  -9.365524680E+01 # Up to here are coefficients from 1000 K to 5000 K
                                  )
             )
gas = ct.Solution('gri30.cti')
gas.add_species(JetA)

### Engine parameters (reference: Thesis Report Development Of A Preliminary Lifing Analysis Tool For The F135-PW-100 Engine by M.H. Jagtenberg)    
overallPressureRatio = 28
BypassRatio = 0.57
FanPressureRatio = 4.7
Fan_inlet_diameter = 1090/1000 # mm to m 
e_inlet = 0.9
e_fan = 0.90
e_LP_compressor = 0.85
e_HP_compressor = 0.85
e_HP_turbine = 0.9
e_LP_turbine = 0.91
e_core_nozzle = 0.99
e_bypass_nozzle = 0.99
combustorPressureLoss = 0.94 # 6% pressure loss 
peakTemperature = 2175 #K
fuel = 'Jet-A(g)'
mass_flow_rate_air = 90.7 #kg/s

# ft to m converter
Altitude = 0.3048*altitude # m

# Lists to store data for plotting
stations = ['FSM', 'INL', 'FAN', 'CMP',
            'COM', 'HTB', 'LTB',
            'CNZ', 'BNZ']

### Free stream condition or flight condition
atmosphere = atm.ATMOSPHERE_1976(Altitude)
P_altitude = atmosphere.P #Pa
T_altitude = atmosphere.T #K
X = 'O2:1 N2:3.76'
gas.TPX = T_altitude, P_altitude, X
V_altitude = Ma*frozenSoundSpeed(gas)
H_altitude = gas.enthalpy_mass # J/Kg
S_altitude = gas.entropy_mass # J/Kg K

print('Freestream Properties :')
print('Freestream pressure at given altitude (P_altitude) {:2.2f} KPA'.format(P_altitude/1000))
print('Freestream Temperature at given infty is (T_altitude) {:2.2f} K'.format(T_altitude)) 
print('Freestream enthalpy at given infty is (H_altitude) {:2.2f} KJ/Kg'.format(H_altitude/1000))
print('Freestream entropy at given infty is (S_altitude) {:2.2f} KJ/Kg K'.format(S_altitude/1000))
print('Freestream velocity at given infty is (V_altitude) {:2.2f} m/s'.format(V_altitude))
print()

# Stagnation flight conditions
H_0a = H_altitude + ((V_altitude**2) / 2) #J/kg
P_0a = aero2564tools.HS(gas, H_0a, S_altitude)
gas.HP = H_0a, P_0a
S_0a = gas.entropy_mass #J/kg K
T_0a= T_altitude * (1 + 0.5 * (e_fan - 1) * Ma**2)

print('Stagnation properties :')
print('Stagnation pressure is (P_0a) {:2.2f} Kpa'.format(P_0a/1000))
print('Stagnation temperature is (T_0a) {:2.2f} K'.format(T_0a))
print('Stagnation enthalpy is (H_0a) {:2.2f} KJ/kg'.format(H_0a/1000))
print('Stagnation entropy is (S_0a) {:2.2f} KJ/kg k'.format(S_0a/1000))
print()

# Determining properties at the inlet exit
H1_inlet_exit = H_altitude + ((V_altitude**2) / 2) # J/kg
gas.HP = H1_inlet_exit, P_0a # Updating state 
S1 = gas.entropy_mass # J/Kg k
P1_inlet_exit = HS(gas, H1_inlet_exit, S1) #Pa
gas.HP = H1_inlet_exit, P1_inlet_exit
T1_inlet_exit = gas.T
gas.SP = S1, P1_inlet_exit # Updating state 
S1_inlet_exit = gas.entropy_mass  # J/Kg k

print('At Inlet exit properties are:')
print('Pressure at inlet exit is (P1_inlet_exit) {:2.2f} KPa'.format(P1_inlet_exit/1000))
print('Temperature at inlet exit is (T1_inlet_exit) {:2.2f} K'.format(T1_inlet_exit))
print('Enthalpy at inlet exit is (H1_inlet_exit) {:2.2f} KJ/Kg'.format(H1_inlet_exit/1000))
print('Entropy at inlet exit is (S1_inlet_exit) {:2.2f} KJ/kg-K'.format(S1_inlet_exit/1000))
print()

# Determining properties at the fan exit 
P2_fan_exit = P1_inlet_exit*FanPressureRatio #Pa
gas.SP = S1_inlet_exit, P2_fan_exit
S2 = gas.entropy_mass #J/kg k
gas.SP = S2, P2_fan_exit
H2 = gas.enthalpy_mass #J/kg
fan_specific_work_output = (H2 - H1_inlet_exit) / e_fan # J/kg
H2_fan_exit = H2 + fan_specific_work_output # J/kg
gas.HP = H2_fan_exit, P2_fan_exit
S2_fan_exit = gas.entropy_mass #J/kg k
T2_fan_exit = gas.T #K
V2_fan_exit = np.sqrt(2 * (H2_fan_exit - H1_inlet_exit))

print('At fan exit properties are')
print('Pressure at fan exit is (P2_fan_exit) {:2.2f} KPa'.format(P2_fan_exit/1000))
print('Temperature at fan exit is (T2_fan_exit) {:2.2f} K'.format(T2_fan_exit))
print('Enthalpy at fan exit is (H2_fan_exit) {:2.2f} KJ/Kg'.format(H2_fan_exit/1000))
print('Entropy at fan exit is (S2_fan_exit) {:2.2f} KJ/Kg K'.format(S2_fan_exit/1000))
print('Velocity at fan exit is (V2_fan_exit) {:2.2f} m/s'.format(V2_fan_exit))
print()

# Determining properties at the compressor exit 
P3_compressor_exit = P1_inlet_exit*overallPressureRatio #Pa
gas.SP = S2_fan_exit, P3_compressor_exit
S3 = gas.entropy_mass #J/kg k
gas.SP = S3, P3_compressor_exit
H3 = gas.enthalpy_mass #J/kg
Compressor_specific_work_output = (H3 - H2_fan_exit) / e_HP_compressor #J/kg
H3_compressor_exit = H3 + Compressor_specific_work_output #J/kg
gas.HP = H3_compressor_exit, P3_compressor_exit
S3_compressor_exit = gas.entropy_mass #J/kg k
T3_compressor_exit = gas.T #K

print('At compressor exit properties are')
print('Pressor at compressor exit is (P3_compressor exit) {:2.2f} Kpa'.format(P3_compressor_exit/1000))
print('Temperature at compressor exit is (T3_compressor_exit){:2.2f} K'.format(T3_compressor_exit))
print('Enthalpy at compressor exit is (H3_compressor_exit){:2.2f} KJ/Kg'.format(H3_compressor_exit/1000))
print('Entropy at compressor exit is (S3_compressor_exit) {:2.2f} KJ/kg K'.format(S3_compressor_exit/1000))
print()

# Determining properties in combustion chamber
T4_combustion = peakTemperature #K
fuelAirRatio = determineFuelOxidiserRatio(gas, fuel, T4_combustion)
mass_flow_rate_fuel = mass_flow_rate_air*fuelAirRatio #Kg/s
Y_dict = gas.mass_fraction_dict()
Y_dict[fuel] = fuelAirRatio
gas.TPY = gas.T, gas.P, Y_dict
H4_combustion = gas.enthalpy_mass #J/kg
P4_combustion = P3_compressor_exit * combustorPressureLoss #Pa
gas.equilibrate('HP')
gas.HP = H4_combustion, P4_combustion
S4_combustion = gas.entropy_mass #J/Kg k

print('Determining properties at combustion chamber') 
print('Fuel air ratio is {:2.3f} '.format(fuelAirRatio))
print('Pressure in combustion chamber is (P4_combustion) {:2.2f} Kpa'.format(P4_combustion/1000))
print('Temperature in combustion chamber is (T4_combustion) {:2.2f} K'.format(T4_combustion))
print('Enthalpy in combustion chamber is (H4_combustion_chamber) {:2.2f} KJ/ kg'.format(H4_combustion/1000))
print('Entropy in combustion chamber is (S4_combustion_chamber) {:2.2f} KJ/kg K'.format(S4_combustion/1000))
print()

# Determining properties high pressure turbine exit 
High_pressure_turbine_specific_work_output = Compressor_specific_work_output / (1+fuelAirRatio) #J/kg
H5_HP_Turbine_exit = H4_combustion - (High_pressure_turbine_specific_work_output/e_HP_turbine) #J/kg
S5 = gas.entropy_mass #J/kg k
gas.HP = H5_HP_Turbine_exit, P4_combustion
P5_HP_Turbine_exit = HS(gas, H5_HP_Turbine_exit, S5, initialGuess=0.1) #Pa
gas.SP = S5, P5_HP_Turbine_exit
T5_HP_Turbine_exit = gas.T #K
gas.HP = H5_HP_Turbine_exit, P5_HP_Turbine_exit
S5_HP_Turbine_exit = gas.entropy_mass #J/Kg K

print('Determining properties at High Pressure Turbine')
print('Pressure at high pressure turbine (P5_HP_Turbine_exit) {:2.2f} Kpa'.format(P5_HP_Turbine_exit/1000))
print('Temperature at the exit of HP Turbine is (T5_HP_Turbine_exit) {:2.2f} K'.format(T5_HP_Turbine_exit))
print('Enthalpy at the exit of HP Turbine is (H5_HP_Turbine_exit) {:2.2f} KJ/kg'.format(H5_HP_Turbine_exit/1000))
print('Entropy at the exit of HP Turbine is (S5_HP_Turbine_exit) {:2.2f} KJ/Kg k'.format(S5_HP_Turbine_exit/1000))
print()

# Determining properties at low pressure turbine exit
Low_pressure_turbine_specific_work_output = fan_specific_work_output*(BypassRatio + 1) / (1+fuelAirRatio) #J/kg
H6_LP_Turbine_exit = H5_HP_Turbine_exit - (Low_pressure_turbine_specific_work_output/e_LP_turbine) #J/kg
S6 = gas.entropy_mass #J/kg k
gas.HP = H6_LP_Turbine_exit, P5_HP_Turbine_exit
P6_LP_Turbine_exit = HS(gas, H6_LP_Turbine_exit, S6, initialGuess=0.1) #Pa
gas.SP = S6, P6_LP_Turbine_exit
T6_LP_Turbine_exit = gas.T #K
gas.HP = H6_LP_Turbine_exit, P6_LP_Turbine_exit
S6_LP_Turbine_exit = gas.entropy_mass # J/kg K

print('Properties at the exit of low pressure turbine')
print('Pressure at low pressure turbine (P6_LP_Turbine_exit) {:2.2f} Kpa'.format(P6_LP_Turbine_exit/1000))
print('Temperature at the exit of HP Turbine is (T6_LP_Turbine_exit) {:2.2f} K'.format(T6_LP_Turbine_exit))
print('Enthalpy at the exit of HP Turbine is (H6_LP_Turbine_exit) {:2.2f} KJ/kg'.format(H6_LP_Turbine_exit/1000))
print('Entropy at the exit of HP Turbine is (S6_LP_Turbine_exit) {:2.2f} KJ/Kg k'.format(S6_LP_Turbine_exit/1000))
print()

# Determining properties at core nozzle exit 
S7_core_nozzle = gas.entropy_mass #J/Kg k
P7_core_nozzle = P_altitude*e_core_nozzle #Pa
gas.SP = S7_core_nozzle, P7_core_nozzle
H7_core_nozzle = gas.enthalpy_mass #J/kg k
V7_core_nozzle = np.sqrt(2*(H6_LP_Turbine_exit - H7_core_nozzle)) #m/s
gas.HP = H7_core_nozzle, P7_core_nozzle
T7_core_nozzle = gas.T #K

print('Properties at core nozzle')
print('Pressure in core nozzle (P7_core_nozzle){:2.2f} Kpa'.format(P7_core_nozzle/1000))
print('Temperature in core nozzle (T7_core_nozzle){:2.2f} K'.format(T7_core_nozzle))
print('Enthalpy at core nozzle (H7_core_nozzle) {:2.2f} KJ/kg'.format(H7_core_nozzle/1000))
print('Entropy at core nozzle (S7_core_nozzle) {:2.2f} KJ/kg k'.format(S7_core_nozzle/1000))
print('Velocity in core nozzle (V7_core_nozzle){:2.2f} m/s'.format(V7_core_nozzle))
print()

# Determining properties at bypass nozzle exit
gas.HPX = H2_fan_exit, P2_fan_exit, X
gas.SP = S2_fan_exit, P2_fan_exit
S8_bypass_nozzle = gas.entropy_mass #J/kg k
P8_bypass_nozzle = P_altitude * e_bypass_nozzle #Pa
gas.SP = S8_bypass_nozzle, P8_bypass_nozzle
H8_bypass_nozzle = gas.enthalpy_mass #J/kg
gas.HP = H8_bypass_nozzle, P8_bypass_nozzle
T8_bypass_nozzle = gas.T #K
V8_bypass_nozzle = np.sqrt(2*(H2_fan_exit - H8_bypass_nozzle)) #m/s
A8_bypass_nozzle = np.pi / 4 * Fan_inlet_diameter**2 #m^2

print('Properties at bypass nozzle')
print('Pressure in bypass nozzle (P8_bypass_nozzle){:2.2f} Kpa'.format(P8_bypass_nozzle/1000))
print('Temperature at bypass nozzle (T8_bypass_nozzle) {:2.2f} K'.format(T8_bypass_nozzle))
print('Enthalpy at bypass nozzle (H8_bypass_nozzle) {:2.2f} KJ/kg k'.format(H8_bypass_nozzle/1000))
print('Entropy at core nozzle (S8_bypass_nozzle){:2.2f} KJ/kg k'.format(S8_bypass_nozzle/1000))
print('Velocity at core nozzle (V8_bypass_nozzle){:2.2f} m/s'.format(V8_bypass_nozzle))
print()

# Evaluating thrust and thrust Specific Fuel Consumption (TSFC)
mass_flow_rate_total = mass_flow_rate_air + mass_flow_rate_fuel #kg/s
F_core = mass_flow_rate_total * V7_core_nozzle + (1 + fuelAirRatio) * (V7_core_nozzle - V_altitude) # Calculate thrust from the core nozzle
F_bypass = mass_flow_rate_air * BypassRatio * V8_bypass_nozzle # Calculate thrust from the bypass nozzle
F_total = F_core + F_bypass # Calculate total thrust
specificThrust = -V_altitude + BypassRatio/(1+BypassRatio)*V8_bypass_nozzle+ (1/(1+BypassRatio))*(1+(fuelAirRatio/overallPressureRatio))*V7_core_nozzle
thrustSpecificFuelConsumption = fuelAirRatio/(1+BypassRatio)/specificThrust # kg/N.s

print('Calculating thrust and specific fuel consumption of thrust: ')
print('Mass flow rate of air is (mass_flow_rate_air) {:2.2f} Kg/s'.format(mass_flow_rate_air))
print('Mass flow rate of fuel is (mass_flow_rate_fuel) {:2.2f} Kg/s'.format(mass_flow_rate_fuel))
print('Total mass flow rate is (mass_flow_rate_total) {:2.2f} Kg/s'.format(mass_flow_rate_total))
print(f'Thrust of the core nozzle: {F_core/1000} kN')
print(f'Thrust of the bypass nozzle: {F_bypass/1000} kN')
print(f'Total Thrust: {F_total/1000} kN')
print('The specific thrust of this engine is {:.1f} m/s and the TSFC is {:.1f} g/kN/s'.format(specificThrust,thrustSpecificFuelConsumption*1e3*1e3))

# Data for plotting
altitudes = [Altitude, Altitude, Altitude, Altitude, Altitude,
             Altitude, Altitude, Altitude, Altitude]
pressures = [P_altitude, P1_inlet_exit, P2_fan_exit, P3_compressor_exit,
             P4_combustion, P5_HP_Turbine_exit, P6_LP_Turbine_exit,
             P7_core_nozzle, P8_bypass_nozzle]
temperatures = [T_altitude, T1_inlet_exit, T2_fan_exit, T3_compressor_exit,
                T4_combustion, T5_HP_Turbine_exit, T6_LP_Turbine_exit,
                T7_core_nozzle, T8_bypass_nozzle]
entropies = [S_altitude, S1_inlet_exit, S2_fan_exit, S3_compressor_exit,
             S4_combustion, S5_HP_Turbine_exit, S6_LP_Turbine_exit,
             S7_core_nozzle, S8_bypass_nozzle]
enthalpies = [H_altitude, H1_inlet_exit, H2_fan_exit, H3_compressor_exit,
              H4_combustion, H5_HP_Turbine_exit, H6_LP_Turbine_exit,
              H7_core_nozzle, H8_bypass_nozzle]

# Plotting the data
plt.figure(figsize=(12, 8))

# Plot Pressure
plt.subplot(2, 2, 1)
plt.plot(stations, pressures, marker='o', color='b')
plt.title(f'Pressure Variation at Altitude {Altitude} m')
plt.xlabel('Stations')
plt.ylabel('Pressure (Pa)')

# Plot Temperature
plt.subplot(2, 2, 2)
plt.plot(stations, temperatures, marker='D', color='g')
plt.title(f'Temperature Variation at Altitude {Altitude} m')
plt.xlabel('Stations')
plt.ylabel('Temperature (K)')

# Plot Entropy
plt.subplot(2, 2, 3)
plt.plot(stations, entropies, marker='s', color='r')
plt.title(f'Entropy Variation at Altitude {Altitude} m')
plt.xlabel('Stations')
plt.ylabel('Entropy (J/kg K)')

# Plot Enthalpy
plt.subplot(2, 2, 4)
plt.plot(stations, enthalpies, marker='*', color='purple')
plt.title(f'Enthalpy Variation at Altitude {Altitude} m')
plt.xlabel('Stations')
plt.ylabel('Enthalpy (J/kg)')

plt.tight_layout()
plt.show()






