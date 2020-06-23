# import os
# import sys
# from datetime import datetime
# import pickle
import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import torch

# sys.path.append('../src')
# import deepymod_torch.VE_params as VE_params
# import deepymod_torch.VE_datagen as VE_datagen
# from deepymod_torch.DeepMod import run_deepmod


# random seeding
np_seed = 4
torch_seed = 0
np.random.seed(np_seed)
torch.manual_seed(torch_seed)

##

# general_path = 'Oscilloscope data CRI electronics analogy/'
# specific_path = 'AWG 7V half sinc KELVIN cap 1000/' # It is precisely here that changes the data we are grabbing to test.
# path = general_path + specific_path
path = '/Users/gert-janboth/Documents/deepmod_maxwell/data/'

# Some of these factors are just for saving at the end but...
# ... input_type is used in recalc after DM
# ... omega is used in time scaling.
# ... mech_model is used to predict coeffs and recover mech params
# input_type = 'Strain'
# mech_model = 'GKM'
# func_desc = 'Half Sinc'
omega = 2*np.pi * 5 * 0.1
# Amp = 7

channel_1_data = np.loadtxt(path+'Channel 1 total voltage.csv', delimiter=',', skiprows=3)
channel_2_data = np.loadtxt(path+'Channel 2 voltage shunt resistor.csv', delimiter=',', skiprows=3)

##

lower = 806
upper = -759

voltage_array = channel_1_data[lower:upper, 1:]
voltage_shunt_array = channel_2_data[lower:upper, 1:]
time_array = channel_1_data[lower:upper, :1]

##

# Maxwell shunt
r_shunt = 10.2 # measured using multimeter
# Kelvin shunt
# r_shunt = 10.2 # measured using multimeter

current_array = voltage_shunt_array/r_shunt

##

# 'normalising'
t_sf = omega/1.2 # Aim for this to be such that the T of the scaled data is a bit less than 2pi
V_sf = 1/np.max(abs(voltage_array))
I_sf = 1/np.max(abs(current_array))
scaled_time_array = time_array*t_sf
scaled_voltage = voltage_array*V_sf
scaled_current = current_array*I_sf

# structuring
target_array = np.concatenate((scaled_voltage, scaled_current), axis=1)

##

# random sampling
number_of_samples = scaled_time_array.size

reordered_row_indices = np.random.permutation(scaled_time_array.size)
reduced_time_array = scaled_time_array[reordered_row_indices, :][:number_of_samples]
reduced_target_array = target_array[reordered_row_indices, :][:number_of_samples]

##

time_tensor = torch.tensor(reduced_time_array, dtype=torch.float32, requires_grad=True)
target_tensor = torch.tensor(reduced_target_array, dtype=torch.float32)




# Deepmod stuff
from deepymod_maxwell import DeepMoD
from deepymod_maxwell.model.func_approx import NN
from deepymod_maxwell.model.library import LibraryMaxwellReal
from deepymod_maxwell.model.constraint import LeastSquares
from deepymod_maxwell.model.sparse_estimators import Clustering, Threshold
from deepymod_maxwell.training import train
from deepymod_maxwell.training.sparsity_scheduler import Periodic

network = NN(1, [30, 30, 30, 30], 2)  # Function approximator
library = LibraryMaxwellReal(3) # Library function
estimator = Threshold(threshold=0.05) # Sparse estimator 
constraint = LeastSquares() # How to constrain
model = DeepMoD(network, library, estimator, constraint) # Putting it all in the model

# Running model
sparsity_scheduler = Periodic(initial_epoch=10000, periodicity=500) # Defining when to apply sparsity
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.999), amsgrad=True) # Defining optimizer
train(model, time_tensor, target_tensor, optimizer, sparsity_scheduler, max_iterations=50000, patience=1000, delta=0.01) # Running
