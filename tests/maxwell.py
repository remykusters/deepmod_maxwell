import sys
import numpy as np
import torch
from deepymod_maxwell.data.maxwell import calculate_strain_stress
from deepymod_maxwell.model.library import auto_deriv

import torch.autograd as auto

np_seed = 2
torch_seed = 0
np.random.seed(np_seed)
torch.manual_seed(torch_seed)

##

input_type = 'Strain'

# For Boltzmann DG, specific model required for calculation of response given manipulation type. Strain -> GMM, Stress -> GKM.
# For odeint method, no need to consider.
# mech_model = 'GMM' 

E = [1, 1, 1]
eta = [0.5, 2.5]

##

omega = 1
Amp = 7
input_expr = lambda t: Amp*np.sin(omega*t)/(omega*t)
d_input_expr = lambda t: (Amp/t)*(np.cos(omega*t) - np.sin(omega*t)/(omega*t))
input_torch_lambda = lambda t: Amp*torch.sin(omega*t)/(omega*t)

##

time_array = np.linspace(10**-10, 10*np.pi/omega, 5000).reshape(-1, 1)

strain_array, stress_array = calculate_strain_stress(input_type, time_array, input_expr, E, eta, D_input_lambda=d_input_expr)

##

# 'normalising'
time_sf = omega/1.2
strain_sf = 1/np.max(abs(strain_array))
stress_sf = 1/np.max(abs(stress_array))
# print(time_sf, strain_sf, stress_sf)

scaled_time_array = time_array*time_sf
scaled_strain_array = strain_array*strain_sf
scaled_stress_array = stress_array*stress_sf
if input_type == 'Strain':
    scaled_input_expr = lambda t: strain_sf*input_expr(t/time_sf)
    scaled_input_torch_lambda = lambda t: strain_sf*input_torch_lambda(t/time_sf)
    scaled_target_array = scaled_stress_array
elif input_type == 'Stress':
    scaled_input_expr = lambda t: stress_sf*input_expr(t/time_sf)
    scaled_input_torch_lambda = lambda t: stress_sf*input_torch_lambda(t/time_sf)
    scaled_target_array = scaled_strain_array

##

number_of_samples = 1000

reordered_row_indices = np.random.permutation(scaled_time_array.size)

reduced_time_array = scaled_time_array[reordered_row_indices, :][:number_of_samples]
reduced_target_array = scaled_target_array[reordered_row_indices, :][:number_of_samples]

##

time_tensor = torch.tensor(reduced_time_array, dtype=torch.float32, requires_grad=True)
target_tensor = torch.tensor(reduced_target_array, dtype=torch.float32)



input_data = scaled_input_torch_lambda(time_tensor)
input_derivs = auto_deriv(time_tensor, input_data, 3)
input_theta = torch.cat((input_data.detach(), input_derivs.detach()), dim=1)


# Deepmod stuff
from deepymod_maxwell import DeepMoD
from deepymod_maxwell.model.func_approx import NN
from deepymod_maxwell.model.library import LibraryMaxwell
from deepymod_maxwell.model.constraint import LeastSquares
from deepymod_maxwell.model.sparse_estimators import Clustering, Threshold
from deepymod_maxwell.training import train
from deepymod_maxwell.training.sparsity_scheduler import Periodic

network = NN(1, [30, 30, 30, 30], 1)  # Function approximator
library = LibraryMaxwell(3, input_theta, 'strain') # Library function
estimator = Threshold() # Sparse estimator 
constraint = LeastSquares() # How to constrain
model = DeepMoD(network, library, estimator, constraint) # Putting it all in the model

# Running model
sparsity_scheduler = Periodic(initial_epoch=20000, periodicity=500) # Defining when to apply sparsity
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.999), amsgrad=True) # Defining optimizer
train(model, time_tensor, target_tensor, optimizer, sparsity_scheduler, max_iterations=50000) # Running
