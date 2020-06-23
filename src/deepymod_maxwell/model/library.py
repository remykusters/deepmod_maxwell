import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations, product
from functools import reduce
from .deepmod import Library


# ==================== Library helper functions =======================

def auto_deriv(data, prediction, max_order):
    '''
    data and prediction must be single columned tensors.
    If it is desired to calculate the derivatives of different predictions wrt different data, this function must be called multiple times.
    This function does not return a column with the zeroth derivative (the prediction).
    '''
    
    # First derivative builds off prediction.
    derivs = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    for _ in range(max_order-1):
        # Higher derivatives chain derivatives from first derivative.
        derivs = torch.cat((derivs, grad(derivs[:, -1:], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]), dim=1)
            
    return derivs

def library_poly(prediction, max_order):
    # Calculate the polynomes of u
    u = torch.ones_like(prediction)
    for order in np.arange(1, max_order+1):
        u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

    return u


def library_deriv(data, prediction, max_order):
    dy = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    time_deriv = dy[:, 0:1]

    if max_order == 0:
        du = torch.ones_like(time_deriv)
    else:
        du = torch.cat((torch.ones_like(time_deriv), dy[:, 1:2]), dim=1)
        if max_order > 1:
            for order in np.arange(1, max_order):
                du = torch.cat((du, grad(du[:, order:order+1], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 1:2]), dim=1)

    return time_deriv, du


# ========================= Actual library functions ========================
class Library1D(Library):
    ''' Calculates library consisting of m-th order polynomials,
        n-th order derivatives and their cross terms.
    '''
    def __init__(self, poly_order, diff_order):
        super().__init__()
        self.poly_order = poly_order
        self.diff_order = diff_order

    def library(self, input):
        prediction, data = input
        poly_list = []
        deriv_list = []
        time_deriv_list = []

        # Creating lists for all outputs
        for output in torch.arange(prediction.shape[1]):
            time_deriv, du = library_deriv(data, prediction[:, output:output+1], self.diff_order)
            u = library_poly(prediction[:, output:output+1], self.poly_order)

            poly_list.append(u)
            deriv_list.append(du)
            time_deriv_list.append(time_deriv)

        samples = time_deriv_list[0].shape[0]
        total_terms = poly_list[0].shape[1] * deriv_list[0].shape[1]

        # Calculating theta
        if len(poly_list) == 1:
            theta = torch.matmul(poly_list[0][:, :, None], deriv_list[0][:, None, :]).view(samples, total_terms)  # If we have a single output, we simply calculate and flatten matrix product between polynomials and derivatives to get library
        else:
            theta_uv = reduce((lambda x, y: (x[:, :, None] @ y[:, None, :]).view(samples, -1)), poly_list)
            theta_dudv = torch.cat([torch.matmul(du[:, :, None], dv[:, None, :]).view(samples, -1)[:, 1:] for du, dv in combinations(deriv_list, 2)], 1)  # calculate all unique combinations of derivatives
            # theta_udu = torch.cat([torch.matmul(u[:, 1:, None], du[:, None, 1:]).view(samples, (poly_list[0].shape[1]-1) * (deriv_list[0].shape[1]-1)) for u, dv in product(poly_list, deriv_list)], 1)  # calculate all unique products of polynomials and derivatives
            # theta = torch.cat([theta_uv, theta_dudv, theta_udu], dim=1)
            theta = torch.cat([theta_uv, theta_dudv], dim=1)

        return time_deriv_list, theta


class Library2D(Library):
    def __init__(self, poly_order, diff_order):
        super().__init__()
        self.poly_order = poly_order
        self.diff_order = diff_order

    def library(self, input):
        '''Constructs a library graph in 1D. Library config is dictionary with required terms. '''
        prediction, data = input
        # Polynomial

        u = torch.ones_like(prediction)
        for order in np.arange(1, self.poly_order+1):
            u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

        # Gradients
        du = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_y = du[:, 2:3]
        du2 = grad(u_x, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_xx = du2[:, 1:2]
        u_xy = du2[:, 2:3]
        u_yy = grad(u_y, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 2:3]

        du = torch.cat((torch.ones_like(u_x), u_x, u_y, u_xx, u_yy, u_xy), dim=1)

        samples = du.shape[0]
        # Bringing it together
        theta = torch.matmul(u[:, :, None], du[:, None, :]).view(samples, -1)

        return [u_t], theta


class LibraryMaxwell(Library):
    def __init__(self, diff_order, input_theta, lib_type):
        super().__init__()
        self.diff_order = diff_order
        self.input_theta = input_theta
        self.lib_type = lib_type

    def library(self, input):  
        prediction, data = input
    
        # Automatic derivatives of response variable 
        output_derivs = auto_deriv(data, prediction, self.diff_order)
        output_theta = torch.cat((prediction, output_derivs), dim=1)
    
        # Identify the manipulation/response as Stress/Strain and organise into returned variables
        if self.lib_type == 'strain':
            strain = self.input_theta
            stress = output_theta
        else: # 'Stress'
            strain = output_theta
            stress = self.input_theta
        
        strain_t = strain[:, 1:2] # Extract the first time derivative of strain
        strain = torch.cat((strain[:, 0:1], strain[:, 2:]), dim=1) # remove this before it gets put into theta
        strain *= -1 # The coefficient of all strain terms will always be negative. rather than hoping deepmod will find these negative terms, we assume the negative factor here and later on DeepMoD will just find positive coefficients
        theta = torch.cat((strain, stress), dim=1) # I have arbitrarily set the convention of making Strain the first columns of data
    
        return [strain_t], theta


class LibraryMaxwell(Library):
    def __init__(self, diff_order, input_theta, lib_type):
        super().__init__()
        self.diff_order = diff_order
        self.input_theta = input_theta
        self.lib_type = lib_type

    def library(self, input):  
        prediction, data = input
    
        # Automatic derivatives of response variable 
        output_derivs = auto_deriv(data, prediction, self.diff_order)
        output_theta = torch.cat((prediction, output_derivs), dim=1)
    
        # Identify the manipulation/response as Stress/Strain and organise into returned variables
        if self.lib_type == 'strain':
            strain = self.input_theta
            stress = output_theta
        else: # 'Stress'
            strain = output_theta
            stress = self.input_theta
        
        strain_t = strain[:, 1:2] # Extract the first time derivative of strain
        strain = torch.cat((strain[:, 0:1], strain[:, 2:]), dim=1) # remove this before it gets put into theta
        strain *= -1 # The coefficient of all strain terms will always be negative. rather than hoping deepmod will find these negative terms, we assume the negative factor here and later on DeepMoD will just find positive coefficients
        theta = torch.cat((strain, stress), dim=1) # I have arbitrarily set the convention of making Strain the first columns of data
    
        return [strain_t], theta

class LibraryMaxwellReal(Library):
    def __init__(self, diff_order):
        super().__init__()
        self.diff_order = diff_order

    def library(self, input):    
        prediction, data = input
        # The first column of prediction is always strain
        strain_derivs = auto_deriv(data, prediction[:, :1], self.diff_order)
        strain_theta = torch.cat((prediction[:, :1], strain_derivs), dim=1)
        
        # The second column is always stress
        stress_derivs = auto_deriv(data, prediction[:, 1:], self.diff_order)
        stress_theta = torch.cat((prediction[:, 1:], stress_derivs), dim=1)
        
        strain_t = strain_theta[:, 1:2] # Extract the first time derivative of strain
        strain_theta = torch.cat((strain_theta[:, 0:1], strain_theta[:, 2:]), dim=1) # remove this before it gets put into theta
        strain_theta *= -1 # The coefficient of all strain terms will always be negative. rather than hoping deepmod will find these negative terms, we assume the negative factor here and later on DeepMoD will just find positive coefficients
        theta = torch.cat((strain_theta, stress_theta), dim=1) # I have arbitrarily set the convention of making Strain the first columns of data
        
        return [strain_t], theta