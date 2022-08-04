import numpy as np
from posterior_helper_functions import *

# Gaussian process prior implementation, given a mu vector and sigma matrix 
def GPP(b,mu,sigma_matrix): 
    # gaussian generalization to higher dimension
    # b = parameter that we're interested in: bin heights 
    
    # log(GPP) = -(b-mu).(sigma^-1)/2.(b-mu)
    logGPP = np.dot(0.5*np.linalg.inv(sigma_matrix), b-mu)
    logGPP = -np.dot(b-mu, logGPP)
    
    return logGPP  

def binned(c,sampleDict,injectionDict,priorDict): 
    
    """
    Implementation of the binned model: a component spin distribution with spin 
    magnitude  as a binned distribution and the cosine of the tilt angle as a binned distirbution.
    
    Parameters
    ----------
    c : `numpy.array`
        array containing hyper-parameter samples in the order:
        
        - [chi bins (without last bin)]
        
        - [cost bins (without last bin)]
        
        - Bq = power law slope of mass distirbution 
        
        
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    priorDict : dict
        Precomputed dictionary containing bounds for the priors on each hyper-parameter
    
    Returns
    -------
    logP : float
        log posterior for the input sample 'c'
    """
    
    chi_x_bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    cost_x_bins = [-1, -0.5, 0, 0.5, 1]
    
    # # Make sure hyper-sample is the right length
    # assert len(c)==5, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    chibin1 = c[0]
    chibin2 = c[1]
    chibin3 = c[2] 
    chibin4 = c[3]
    costbin1 = c[4]
    costbin2 = c[5]
    costbin3 = c[6]
    Bq = c[7]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    chi_last_bin = 1 - (chibin1 + chibin2 + chibin3 + chibin4)
    cost_last_bin = 1 - (costbin1 + costbin2 + costbin3)
    
    chi_bins = [chibin1, chibin2, chibin3, chibin4, chi_last_bin]
    cost_bins = [costbin1, costbin2, costbin3, cost_last_bin]

    # Gaussian processed prior, reject samples past boundaries
    if any(x > 1 for x in chi_bins) or any(x < 0 for x in chi_bins) or any(x > 1 for x in cost_bins) or any(x < 0 for x in cost_bins):
        return -np.inf
    else:
        logP = 0.
        
        # -- Gaussian process prior -- 
        sigma_matrix_chi = np.zeros((5,5))
        sigma_matrix_cost = np.zeros((4,4))

        # squared exponential kernel - sigma matrix of the form: sigma_ij = Lambda * exp(-(x_i - x_j)^2/(2 alpha^2))
        Lambda_chi = 15
        alpha_chi = 0.2 # approx bin width
        for i in np.arange(len(chi_x_bins)-1): 
            xi = (chi_x_bins[i+1]-chi_x_bins[i])/2.0 # center point of bin
            for j in np.arange(len(chi_x_bins)-1): 
                xj = (chi_x_bins[j+1] + chi_x_bins[j])/2.0
                if i==j:
                    sigma_matrix_chi[i,j] = Lambda_chi**2.
                else:
                    sigma_matrix_chi[i,j] = (Lambda_chi**2.)*np.exp((-(xi-xj)**2)/(2.0*alpha_chi**2))

        Lambda_cost = 7.5
        alpha_cost = 0.5 # approx bin width
        for i in np.arange(len(cost_x_bins)-1): 
            xi = (cost_x_bins[i+1]-cost_x_bins[i])/2.0 # center point of bin
            for j in np.arange(len(cost_x_bins)-1): 
                xj = (cost_x_bins[j+1] + cost_x_bins[j])/2.0
                if i==j:
                    sigma_matrix_cost[i,j] = Lambda_cost**2.
                else:
                    sigma_matrix_cost[i,j] = (Lambda_cost**2.)*np.exp((-(xi-xj)**2)/(2.0*alpha_cost**2))


        print('\nsigma matrix chi:', sigma_matrix_chi)
        print('sigma matrix cost:', sigma_matrix_cost)

        # Apply Gaussian process prior
        # chi - mean = 1 (corresponding to flat dist) bc width = 1
        mu_chi = np.ones(len(chi_bins))
        
        bin_heights_chi = np.asarray([f/(chi_x_bins[1]-chi_x_bins[0]) for f,bound in zip(chi_bins,chi_x_bins)])
        logP += GPP(bin_heights_chi,mu_chi,sigma_matrix_chi)
        # cost - mean = 1/2 (corresponding to flat dist) bc width = 2
        mu_cost = 0.5*np.ones(len(cost_bins))
        bin_heights_cost = np.asarray([f/(cost_x_bins[1]-cost_x_bins[0]) for f,bound in zip(cost_bins,cost_x_bins)])
        logP += GPP(bin_heights_cost,mu_cost,sigma_matrix_cost)




        # Prior on Bq - gaussian centered at 0 with sigma=3
        logP -= (Bq**2)/18. 

        # --- Selection effects --- 

        # Unpack injections
        chi1_det = injectionDict['a1']
        chi2_det = injectionDict['a2']
        cost1_det = injectionDict['cost1']
        cost2_det = injectionDict['cost2']
        m1_det = injectionDict['m1']
        m2_det = injectionDict['m2']
        z_det = injectionDict['z']
        dVdz_det = injectionDict['dVdz']

        # Draw probability for component spins, masses, + redshift
        p_draw = injectionDict['p_draw_a1a2cost1cost2']*injectionDict['p_draw_m1m2z']

        # Detected spins
        p_chi1_det = calculate_Bins(chi1_det, chi_bins, chi_x_bins)
        p_chi2_det = calculate_Bins(chi2_det, chi_bins, chi_x_bins)
        p_cost1_det = calculate_Bins(cost1_det, cost_bins, cost_x_bins)
        p_cost2_det = calculate_Bins(cost2_det, cost_bins, cost_x_bins)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det

        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq)
        pdet_z = p_astro_z(z_det, dVdz_det)

        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw

        if np.max(det_weights)==0:
            return -np.inf

        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Nsamp<=4*nEvents:
            return -np.inf

        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff

        # --- Loop across BBH events ---

        for event in sampleDict:

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

            # Sum over probabilities to get the marginalized likelihood for this event
            pEvidence = (1.0/nSamples)*np.sum(pSpins*m1_m2_z_prior_ratio/spin_PE_prior)

            # Add to our running total
            logP += np.log(pEvidence)

        if logP!=logP:
            return -np.inf

        else:
            return logP    
        
    
def oldBinned(c,sampleDict,injectionDict,priorDict): 
    
    """
    Implementation of the binned model: a component spin distribution with spin 
    magnitude  as a binned distribution and the cosine of the tilt angle as a binned distirbution.
    
    Parameters
    ----------
    c : `numpy.array`
        array containing hyper-parameter samples in the order:
        
        - [chi bins (without last bin)]
        
        - [cost bins (without last bin)]
        
        - Bq = power law slope of mass distirbution 
        
        
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    priorDict : dict
        Precomputed dictionary containing bounds for the priors on each hyper-parameter
    
    Returns
    -------
    logP : float
        log posterior for the input sample 'c'
    """
    
    chi_x_bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    cost_x_bins = [-1, -0.5, 0, 0.5, 1]
    
    # # Make sure hyper-sample is the right length
    # assert len(c)==5, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    chibin1 = c[0]
    chibin2 = c[1]
    chibin3 = c[2] 
    chibin4 = c[3]
    costbin1 = c[4]
    costbin2 = c[5]
    costbin3 = c[6]
    Bq = c[7]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    chi_last_bin = 1 - (chibin1 + chibin2 + chibin3 + chibin4)
    cost_last_bin = 1 - (costbin1 + costbin2 + costbin3)
    
    chi_bins = [chibin1, chibin2, chibin3, chibin4, chi_last_bin]
    cost_bins = [costbin1, costbin2, costbin3, cost_last_bin]
    
    # Flat priors, reject samples past boundaries
    if any(x > 1 for x in chi_bins) or any(x < 0 for x in chi_bins) or any(x > 1 for x in cost_bins) or any(x < 0 for x in cost_bins):
        return -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
#         # TODO: LEFT OFF HERE
        # Initialize log-posterior
        logP = 0.
        
        # Prior on Bq - gaussian centered at 0 with sigma=3
        logP -= (Bq**2)/18. 
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = injectionDict['a1']
        chi2_det = injectionDict['a2']
        cost1_det = injectionDict['cost1']
        cost2_det = injectionDict['cost2']
        m1_det = injectionDict['m1']
        m2_det = injectionDict['m2']
        z_det = injectionDict['z']
        dVdz_det = injectionDict['dVdz']
        
        # Draw probability for component spins, masses, + redshift
        p_draw = injectionDict['p_draw_a1a2cost1cost2']*injectionDict['p_draw_m1m2z']
        
        # Detected spins
        p_chi1_det = calculate_Bins(chi1_det, chi_bins, chi_x_bins)
        p_chi2_det = calculate_Bins(chi2_det, chi_bins, chi_x_bins)
        p_cost1_det = calculate_Bins(cost1_det, cost_bins, cost_x_bins)
        p_cost2_det = calculate_Bins(cost2_det, cost_bins, cost_x_bins)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq)
        pdet_z = p_astro_z(z_det, dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Nsamp<=4*nEvents:
            return -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff
        
        # --- Loop across BBH events ---
    
        for event in sampleDict:

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
            
            # Sum over probabilities to get the marginalized likelihood for this event
            pEvidence = (1.0/nSamples)*np.sum(pSpins*m1_m2_z_prior_ratio/spin_PE_prior)

            # Add to our running total
            logP += np.log(pEvidence)

        if logP!=logP:
            return -np.inf

        else:
            return logP    
    
    
def gaussian(c,sampleDict,injectionDict,priorDict): 
    
    """
    Implementation of the Gaussian model: a component spin distribution with spin 
    magnitude  as a Gaussian distribution and the cosine of the tilt angle as a Gaussian distirbution.
    
    Parameters
    ----------
    c : `numpy.array`
        array containing hyper-parameter samples in the order: [ mu_chi, sigma_chi, mu_cost, sigma_cost ] where 
        
        - mu_chi = mean of spin magnitude beta distribution
        
        - sigma_chi = std. dev. of spin magnitude beta distribution 
        
        - mu_cost = mean of cos tilt angle distribution
        
        - sigma_cost = std. dev. of cos tilt angle distribution
        
        - Bq = power law slope of mass distirbution 
        
        
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    priorDict : dict
        Precomputed dictionary containing bounds for the priors on each hyper-parameter
    
    Returns
    -------
    logP : float
        log posterior for the input sample 'c'
    """
    
    # Make sure hyper-sample is the right length
    assert len(c)==5, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu_chi = c[0]
    sigma_chi = c[1]
    mu_cost = c[2] 
    sigma_cost = c[3]
    Bq = c[4]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu_chi < priorDict['mu_chi'][0] or mu_chi > priorDict['mu_chi'][1]:
        return -np.inf
    elif sigma_chi < priorDict['sigma_chi'][0] or sigma_chi > priorDict['sigma_chi'][1]:
        return -np.inf
    elif mu_cost < priorDict['mu_cost'][0] or mu_cost > priorDict['mu_cost'][1]:
        return -np.inf
    elif sigma_cost < priorDict['sigma_cost'][0] or sigma_cost > priorDict['sigma_cost'][1]:
        return -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
#         # TODO: LEFT OFF HERE
        # Initialize log-posterior
        logP = 0.
        
#         # Translate mu_chi and sigma_chi to beta function parameters a and b 
#         # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
#         a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)
        
#         # Impose cut on a and b: must be greater then or equal to 1 in order
#         # for distribution to go to 0 at chi=0 and chi=1
#         if a<=1. or b<=1.: 
#             return -np.inf
        
#         # To match the gwtc-3 catalog, we want our hyper prior uniform in sigma^2_chi 
#         # not sigma_chi
#         logP += np.log(sigma_chi)
        
        # Prior on Bq - gaussian centered at 0 with sigma=3
        logP -= (Bq**2)/18. 
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = injectionDict['a1']
        chi2_det = injectionDict['a2']
        cost1_det = injectionDict['cost1']
        cost2_det = injectionDict['cost2']
        m1_det = injectionDict['m1']
        m2_det = injectionDict['m2']
        z_det = injectionDict['z']
        dVdz_det = injectionDict['dVdz']
        
        # Draw probability for component spins, masses, + redshift
        p_draw = injectionDict['p_draw_a1a2cost1cost2']*injectionDict['p_draw_m1m2z']
        
        # Detected spins
        p_chi1_det = calculate_Gaussian_1D(chi1_det, mu_chi, sigma_chi, 0, 1)
        p_chi2_det = calculate_Gaussian_1D(chi2_det, mu_chi, sigma_chi, 0, 1)
        p_cost1_det = calculate_Gaussian_1D(cost1_det, mu_cost, sigma_cost, -1, 1)
        p_cost2_det = calculate_Gaussian_1D(cost2_det, mu_cost, sigma_cost, -1, 1)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq)
        pdet_z = p_astro_z(z_det, dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Nsamp<=4*nEvents:
            return -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff
        
        # --- Loop across BBH events ---
        for event in sampleDict:

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
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = calculate_Gaussian_1D(chi1_samples, mu_chi, sigma_chi, 0, 1)
            p_chi2 = calculate_Gaussian_1D(chi2_samples, mu_chi, sigma_chi, 0, 1)
            p_cost1 = calculate_Gaussian_1D(cost1_samples, mu_cost, sigma_cost, -1, 1)
            p_cost2 = calculate_Gaussian_1D(cost2_samples, mu_cost, sigma_cost, -1, 1)
            
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
            
            # Sum over probabilities to get the marginalized likelihood for this event
            pEvidence = (1.0/nSamples)*np.sum(pSpins*m1_m2_z_prior_ratio/spin_PE_prior)

            # Add to our running total
            logP += np.log(pEvidence)

        if logP!=logP:
            return -np.inf

        else:
            return logP
        
        
        
def betaPlusMixture(c,sampleDict,injectionDict,priorDict): 
    
    """
    Implementation of the Beta+Mixture model: a component spin distribution with spin magnitude 
    as a beta distribution and the cosine of the tilt angle as a mixture of aligned + isotropic, 
    for inference within `emcee`.
    
    Parameters
    ----------
    c : `numpy.array`
        array containing hyper-parameter samples in the order: [ mu_chi, sigma_chi, MF_cost, 
        sigma_cost, Bq ] where 
        
        - mu_chi = mean of spin magnitude beta distribution
        
        - sigma_chi = std. dev. of spin magnitude beta distribution 
        
        - MF_cost = mixing fraction in aligned spin subpopulation for cos tilt angle distribution
        
        - sigma_cost = std. dev. of aligned spin subpopulation for cos tilt angle distribution
        
        - Bq = power law slope of the mass ratio distribution    
        
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    priorDict : dict
        Precomputed dictionary containing bounds for the priors on each hyper-parameter
    
    Returns
    -------
    logP : float
        log posterior for the input sample 'c'
    """
    
    # Make sure hyper-sample is the right length
    assert len(c)==5, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu_chi = c[0]
    sigma_chi = c[1]
    MF_cost = c[2] 
    sigma_cost = c[3]
    Bq = c[4]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu_chi < priorDict['mu_chi'][0] or mu_chi > priorDict['mu_chi'][1]:
        return -np.inf
    elif sigma_chi < priorDict['sigma_chi'][0] or sigma_chi > priorDict['sigma_chi'][1]:
        return -np.inf
    elif MF_cost < priorDict['MF_cost'][0] or MF_cost > priorDict['MF_cost'][1]:
        return -np.inf
    elif sigma_cost < priorDict['sigma_cost'][0] or sigma_cost > priorDict['sigma_cost'][1]:
        return -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
        
        # Initialize log-posterior
        logP = 0.
        
        # Translate mu_chi and sigma_chi to beta function parameters a and b 
        # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)
        
        # Impose cut on a and b: must be greater then or equal to 1 in order
        # for distribution to go to 0 at chi=0 and chi=1
        if a<=1. or b<=1.: 
            return -np.inf
        
        # To match the gwtc-3 catalog, we want our hyper prior uniform in sigma^2_chi 
        # not sigma_chi
        logP += np.log(sigma_chi)
        
        # Prior on Bq - gaussian centered at 0 with sigma=3
        logP -= (Bq**2)/18. 
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = injectionDict['a1']
        chi2_det = injectionDict['a2']
        cost1_det = injectionDict['cost1']
        cost2_det = injectionDict['cost2']
        m1_det = injectionDict['m1']
        m2_det = injectionDict['m2']
        z_det = injectionDict['z']
        dVdz_det = injectionDict['dVdz']
        
        # Draw probability for component spins, masses, + redshift
        p_draw = injectionDict['p_draw_a1a2cost1cost2']*injectionDict['p_draw_m1m2z']
        
        # Detected spins
        p_chi1_det = betaDistribution(chi1_det, a, b)
        p_chi2_det = betaDistribution(chi2_det, a, b)
        p_cost1_det = calculate_Gaussian_Mixture_1D(cost1_det, 1, sigma_cost, MF_cost, -1, 1.)
        p_cost2_det = calculate_Gaussian_Mixture_1D(cost2_det, 1, sigma_cost, MF_cost, -1, 1.)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq)
        pdet_z = p_astro_z(z_det, dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Nsamp<=4*nEvents:
            return -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff
        
        # --- Loop across BBH events ---
        for event in sampleDict:

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
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = betaDistribution(chi1_samples, a, b)
            p_chi2 = betaDistribution(chi2_samples, a, b)
            p_cost1 = calculate_Gaussian_Mixture_1D(cost1_samples, 1, sigma_cost, MF_cost, -1, 1.)
            p_cost2 = calculate_Gaussian_Mixture_1D(cost2_samples, 1, sigma_cost, MF_cost, -1, 1.)
            
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
            
            # Sum over probabilities to get the marginalized likelihood for this event
            pEvidence = (1.0/nSamples)*np.sum(pSpins*m1_m2_z_prior_ratio/spin_PE_prior)

            # Add to our running total
            logP += np.log(pEvidence)

        if logP!=logP:
            return -np.inf

        else:
            return logP   
        
        
        
def betaPlusTruncatedMixture(c,sampleDict,injectionDict,priorDict): 
    
    """
    Implementation of the Beta+TruncatedMixture model: a component spin distribution with spin 
    magnitude  as a beta distribution and the cosine of the tilt angle as a mixture of aligned + isotropic 
    with a lower truncation, for inference within `emcee`.
    
    Parameters
    ----------
    c : `numpy.array`
        array containing hyper-parameter samples in the order: [ mu_chi, sigma_chi, MF_cost, 
        sigma_cost, cost_min, Bq ] where 
        
        - mu_chi = mean of spin magnitude beta distribution
        
        - sigma_chi = std. dev. of spin magnitude beta distribution 
        
        - MF_cost = mixing fraction in aligned spin subpopulation for cos tilt angle distribution
        
        - sigma_cost = std. dev. of aligned spin subpopulation for cos tilt angle distribution
        
        - cost_min = lower truncation bound on the cosine tilt angle distribution
        
        - Bq = power law slope of the mass ratio distribution    
        
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    priorDict : dict
        Precomputed dictionary containing bounds for the priors on each hyper-parameter
    
    Returns
    -------
    logP : float
        log posterior for the input sample 'c'
    """
    
    # Make sure hyper-sample is the right length
    assert len(c)==6, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu_chi = c[0]
    sigma_chi = c[1]
    MF_cost = c[2] 
    sigma_cost = c[3]
    cost_min = c[4]
    Bq = c[5]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu_chi < priorDict['mu_chi'][0] or mu_chi > priorDict['mu_chi'][1]:
        return -np.inf
    elif sigma_chi < priorDict['sigma_chi'][0] or sigma_chi > priorDict['sigma_chi'][1]:
        return -np.inf
    elif MF_cost < priorDict['MF_cost'][0] or MF_cost > priorDict['MF_cost'][1]:
        return -np.inf
    elif sigma_cost < priorDict['sigma_cost'][0] or sigma_cost > priorDict['sigma_cost'][1]:
        return -np.inf
    elif cost_min < priorDict['cost_min'][0] or cost_min > priorDict['cost_min'][1]:
        return -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
        
        # Initialize log-posterior
        logP = 0.
        
        # Translate mu_chi and sigma_chi to beta function parameters a and b 
        # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)
        
        # Impose cut on a and b: must be greater then or equal to 1 in order
        # for distribution to go to 0 at chi=0 and chi=1
        if a<=1. or b<=1.: 
            return -np.inf
        
        # To match the gwtc-3 catalog, we want our hyper prior uniform in sigma^2_chi 
        # not sigma_chi
        logP += np.log(sigma_chi)
        
        # Prior on Bq - gaussian centered at 0 with sigma=3
        logP -= (Bq**2)/18. 
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = injectionDict['a1']
        chi2_det = injectionDict['a2']
        cost1_det = injectionDict['cost1']
        cost2_det = injectionDict['cost2']
        m1_det = injectionDict['m1']
        m2_det = injectionDict['m2']
        z_det = injectionDict['z']
        dVdz_det = injectionDict['dVdz']
        
        # Draw probability for component spins, masses, + redshift
        p_draw = injectionDict['p_draw_a1a2cost1cost2']*injectionDict['p_draw_m1m2z']
        
        # Detected spins
        p_chi1_det = betaDistribution(chi1_det, a, b)
        p_chi2_det = betaDistribution(chi2_det, a, b)
        p_cost1_det = calculate_Gaussian_Mixture_1D(cost1_det, 1, sigma_cost, MF_cost, cost_min, 1.)
        p_cost2_det = calculate_Gaussian_Mixture_1D(cost2_det, 1, sigma_cost, MF_cost, cost_min, 1.)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq)
        pdet_z = p_astro_z(z_det, dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Nsamp<=4*nEvents:
            return -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff
        
        # --- Loop across BBH events ---
        for event in sampleDict:

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
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = betaDistribution(chi1_samples, a, b)
            p_chi2 = betaDistribution(chi2_samples, a, b)
            p_cost1 = calculate_Gaussian_Mixture_1D(cost1_samples, 1, sigma_cost, MF_cost, cost_min, 1.)
            p_cost2 = calculate_Gaussian_Mixture_1D(cost2_samples, 1, sigma_cost, MF_cost, cost_min, 1.)
            
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
            
            # Sum over probabilities to get the marginalized likelihood for this event
            pEvidence = (1.0/nSamples)*np.sum(pSpins*m1_m2_z_prior_ratio/spin_PE_prior)

            # Add to our running total
            logP += np.log(pEvidence)

        if logP!=logP:
            return -np.inf

        else:
            return logP