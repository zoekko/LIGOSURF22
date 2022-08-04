import numpy as np
import sys
from scipy.special import erf
from scipy.special import beta
import pickle
import json 

import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import Planck15

from posterior_helper_functions import * 

pop = 'pop3'
num_injections = '70'
model = 'binned'
fname = 'binned'

chi_x_bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
cost_x_bins = [-1, -0.5, 0, 0.5, 1]

"""
Function to do posterior reweighting for betaPlusMixtureModel 
    sampleDict = dictionary containing individual event samples
    hyperPEDict = dictionary containing hyperparemeter samples: from json file that emcee outputs: 
        samples are the individual curves from the final trace plots (less than the actual number of samples
        for each ind event)
"""
def pop_reweight(sampleDict, hyperPEDict): 
    
    # Number of hyperparameter samples
    nHyperPESamps = len(hyperPEDict['chibin1']['processed'])
    
    # dict in which to put reweighted individual event samples
    sampleDict_rw = {}
    
    # cycle through events
    for k, event in enumerate(sampleDict): 
        print(f"event {k+1} of {len(sampleDict)}: {event}")
        
        # Unpack posterior samples for this event
        chi1_samples = sampleDict[event]['a1']
        chi2_samples =  sampleDict[event]['a2']
        cost1_samples = sampleDict[event]['cost1']
        cost2_samples = sampleDict[event]['cost2']
        m1_samples = sampleDict[event]['m1']
        m2_samples = sampleDict[event]['m2']
        z_samples = sampleDict[event]['z']
        z_prior_samples = sampleDict[event]['z_prior']
        dVdz_samples = sampleDict[event]['dVc_dz']

        # indices corresponding to each sample for these events (will be used below in the for loop)
        nSamples = len(chi1_samples)
        indices = np.arange(nSamples)
        
        # arrays in which to store reweighted samples for this event
        new_chi1_samps = np.zeros(nHyperPESamps)
        new_chi2_samps = np.zeros(nHyperPESamps)
        new_cost1_samps = np.zeros(nHyperPESamps)
        new_cost2_samps = np.zeros(nHyperPESamps)
        new_mass1_samps = np.zeros(nHyperPESamps)
        new_mass2_samps = np.zeros(nHyperPESamps)
        
        # cycle through hyper PE samps
        for i in range(nHyperPESamps):
            '''iterating through each curve of the trace plot'''
            
            # Fetch i^th hyper PE sample: 
            chibin1 = hyperPEDict['chibin1']['processed'][i]
            chibin2 = hyperPEDict['chibin2']['processed'][i]
            chibin3 = hyperPEDict['chibin3']['processed'][i]
            chibin4 = hyperPEDict['chibin4']['processed'][i]
            chibin5 = chibin1 + chibin2 + chibin3 + chibin4
            chibin5 = 1 - chibin5
            chi_bins = [chibin1, chibin2, chibin3, chibin4, chibin5]

            costbin1 = hyperPEDict['costbin1']['processed'][i]
            costbin2 = hyperPEDict['costbin2']['processed'][i]
            costbin3 = hyperPEDict['costbin3']['processed'][i]
            costbin4 = costbin1 + costbin2 + costbin3
            costbin4 = 1 - costbin4
            cost_bins = [costbin1, costbin2, costbin3, costbin4]
            
            Bq = hyperPEDict['Bq']['processed'][i]
            
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = calculate_Bins(chi1_samples, chi_bins, chi_x_bins)
            p_chi2 = calculate_Bins(chi2_samples, chi_bins, chi_x_bins)
            p_cost1 = calculate_Bins(cost1_samples, cost_bins, cost_x_bins)
            p_cost2 = calculate_Bins(cost2_samples, cost_bins, cost_x_bins)
            
            # Pop dist for all four params combined is a product of each four individual dists
            pSpins = p_chi1*p_chi2*p_cost1*p_cost2
            
            # PE priors for chi_i and cost_i are all uniform, so we set them to unity here
            nSamples = pSpins.size
            spin_PE_prior = np.ones(nSamples)
            
            # Need to reweight by astrophysical priors on m1, m2, z ...
            # - p(m1)*p(m2)
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, bq=Bq)
            old_m1_m2_prior = np.ones(nSamples) # PE prior on masses is uniform in component masses
            # - p(z)
            p_astro_redshift = p_astro_z(z_samples, dVdz_samples)
            # - For full m1, m2, z prior reweighting: 
            m1_m2_z_prior_ratio = (p_astro_m1_m2/old_m1_m2_prior)*(p_astro_redshift/z_prior_samples)
            
            # calculate weights for this hyper parameter
            weights = pSpins*m1_m2_z_prior_ratio/spin_PE_prior
            weights = weights/np.sum(weights)
            
            # select a random sample from the event posterior subject to these weights
            j = np.random.choice(indices, p=weights)
            
            # populate the new sample arrays with this random sample
            new_chi1_samps[i] = chi1_samples[j]
            new_chi2_samps[i] = chi2_samples[j]
            new_cost1_samps[i] = cost1_samples[j]
            new_cost2_samps[i] = cost2_samples[j]
            new_mass1_samps[i] = m1_samples[j]
            new_mass2_samps[i] = m2_samples[j]
        
        # Add into reweighted sampleDict
        sampleDict_rw[event] = {
            'chi1':new_chi1_samps,
            'chi2':new_chi2_samps,
            'cost1':new_cost1_samps,
            'cost2':new_cost2_samps,
            'm1':new_mass1_samps,
            'm2':new_mass2_samps
        }

    return sampleDict_rw


"""
Actually loading and running pop reweighting
"""

'''TODO: rename directories'''

if __name__=="__main__":
    
    # Load dict with individual event PE samples: 
    sampleDict_fname = "/home/zoe.ko/LIGOSURF22/input/" + num_injections + "injections/sampleDict_" + pop
    
    with open(f'{sampleDict_fname}.json', 'r') as f:
        sampleDict = json.load(f)
    
    # Load population parameter PE samps
    hyperPEDict_fname = "/home/zoe.ko/LIGOSURF22/data/" + num_injections + "injections/" + num_injections + model + "/" + num_injections + pop + "_" + fname
    with open(f'{hyperPEDict_fname}.json', 'r') as f:
        hyperPEDict = json.load(f)
    
    # Run reweighting for default model
    print("Running reweighting ... ")
    sampleDict_rw = pop_reweight(sampleDict, hyperPEDict)   
    
    # Save results
    savename = sampleDict_fname = "/home/zoe.ko/LIGOSURF22/data/" + num_injections + "injections/" + num_injections + model + "/" + pop + "rw_sampleDict.pickle"
    with open(savename, 'wb') as f:
        pickle.dump(sampleDict_rw,f)