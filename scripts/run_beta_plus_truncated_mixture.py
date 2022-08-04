import numpy as np
import glob
import emcee as mc
import json
import sys
from posterior_helper_functions import draw_initial_walkers_uniform, merge_dicts
from posteriors import betaPlusTruncatedMixture
from postprocessing import processEmceeChain

pop = '3'
numinjections = '300'

# set seed for reproducibility (number chosen arbitrarily)
np.random.seed(2647)

"""
Definitions and loading data
"""

# Model 
model = numinjections + "pop" + pop + "_component_spin_betaPlusTruncatedMixture"

# Repository root 
froot = "/home/zoe.ko/LIGOSURF22/"

# Define emcee parameters
nWalkers = 20       # number of walkers 
dim = 6             # dimension of parameter space (number hyper params)
nSteps = 35000      # number of steps for chain

# Set prior bounds (where applicable, same as Table XII in https://arxiv.org/pdf/2111.03634.pdf)
priorDict = {
    'mu_chi':(0., 1.),
    'sigma_chi':(0.07, 0.5), # the sigma^2 bounds are [0.005, 0.25], so sigma goes from [sqrt(0.005), 4]  
    'MF_cost':(0., 1.),
    'sigma_cost':(0.1, 4.), 
    'cost_min':(-1., 1.)
}



# Load sampleDict
if numinjections == '300':
    f = open(f'{froot}input/10injections/sampleDict_pop{pop}.json')
    sampleDict_10 = json.load(f) 
    f = open(f'{froot}input/70injections/sampleDict_pop{pop}.json')
    sampleDict_70 = json.load(f) 
    f = open(f'{froot}input/100injections/sampleDict_pop{pop}.json')
    sampleDict_100 = json.load(f)
    f = open(f'{froot}input/120injections/sampleDict_pop{pop}.json')
    sampleDict_120 = json.load(f)
    sampleDict = merge_dicts([sampleDict_10, sampleDict_70, sampleDict_100, sampleDict_120])
elif numinjections == '150':
    f = open(f'{froot}input/100injections/sampleDict_pop{pop}.json')
    sampleDict_100 = json.load(f)
    f = open(f'{froot}input/70injections/sampleDict_pop{pop}.json')
    sampleDict_70 = json.load(f) 
    sampleDict = merge_dicts([sampleDict_100, sampleDict_70])
    length = len(sampleDict)
    to_pop = length - 150
    keys = list(sampleDict.keys())
    keys_to_pop = np.random.choice(keys, size=to_pop, replace=False)
    
    for key in keys_to_pop:
        sampleDict.pop(key)
else:
    f = open(f'{froot}input/{numinjections}injections/sampleDict_pop{pop}.json')
    sampleDict = json.load(f) 

# Load injectionDict (for selection effects)
injectionDict = np.load(froot+"input/injectionDict_FAR_1_in_1.pickle", allow_pickle=True)

# Will save emcee chains temporarily in the .tmp folder in this directory
output_folder_tmp = froot+"scripts/.tmp"+numinjections + "_" + pop+"/"
output_tmp = output_folder_tmp+model


"""
Initializing emcee walkers or picking up where an old chain left off
"""

# Search for existing chains
old_chains = np.sort(glob.glob("{0}_r??.npy".format(output_tmp)))

# If no chain already exists, begin a new one
if len(old_chains)==0:
    
    print('\nNo old chains found, generating initial walkers ... ')

    run_version = 0

    # Initialize walkers
    initial_mu_chis = draw_initial_walkers_uniform(nWalkers, (0.2,0.4))
    initial_sigma_chis = draw_initial_walkers_uniform(nWalkers, (0.17,0.25))
    initial_MFs = draw_initial_walkers_uniform(nWalkers, (0.5,1.0))
    initial_sigma_costs = draw_initial_walkers_uniform(nWalkers, (0.1, 2.))
    initial_min_costs = draw_initial_walkers_uniform(nWalkers, (-0.5, 0))
    initial_Bqs = np.random.normal(loc=0, scale=3, size=nWalkers)

    
    # Put together all initial walkers into a single array
    initial_walkers = np.transpose(
        [initial_mu_chis, initial_sigma_chis, initial_MFs, initial_sigma_costs, initial_min_costs, initial_Bqs]
    )
            
# Otherwise resume existing chain
else:
    
    print('\nOld chains found, loading and picking up where they left off ... ' )
    
    # Load existing file and iterate run version
    old_chain = np.concatenate([np.load(chain) for chain in old_chains], axis=1)
    run_version = int(old_chains[-1][-6:-4])+1

    # Strip off any trailing zeros due to incomplete run
    goodInds = np.where(old_chain[0,:,0]!=0.0)[0]
    old_chain = old_chain[:,goodInds,:]

    # Initialize new walker locations to final locations from old chain
    initial_walkers = old_chain[:,-1,:]
    
    # Figure out how many more steps we need to take 
    nSteps = nSteps - old_chain.shape[1]
    
        
print('Initial walkers:')
print(initial_walkers)


"""
Launching emcee
"""

if nSteps>0: # if the run hasn't already finished

    assert dim==initial_walkers.shape[1], "'dim' = wrong number of dimensions for 'initial_walkers'"

    print(f'\nLaunching emcee with {dim} hyper-parameters, {nSteps} steps, and {nWalkers} walkers ...')

    sampler = mc.EnsembleSampler(
        nWalkers,
        dim,
        betaPlusTruncatedMixture, # model in posteriors.py
        args=[sampleDict,injectionDict,priorDict], # arguments passed to betaPlusMixture
        threads=16
    )

    print('\nRunning emcee ... ')

    for i,result in enumerate(sampler.sample(initial_walkers,iterations=nSteps)):

        # Save every 10 iterations
        if i%10==0:
            np.save("{0}_r{1:02d}.npy".format(output_tmp,run_version),sampler.chain)

        # Print progress every 100 iterations
        if i%100==0:
            print(f'On step {i} of {nSteps}', end='\r')

    # Save raw output chains
    np.save("{0}_r{1:02d}.npy".format(output_tmp,run_version),sampler.chain)


"""
Running post processing and saving results
"""

print('\nDoing post processing ...')

if nSteps>0: 

    # If this is the only run, just process this one directly 
    if run_version==0:
        chainRaw = sampler.chain

    # otherwise, put chains from all previous runs together 
    else:
        previous_chains = [np.load(chain) for chain in old_chains]
        previous_chains.append(sampler.chain)
        chainRaw = np.concatenate(previous_chains, axis=1)

else: 
    chainRaw = old_chain

# Run post-processing
chainDownsampled = processEmceeChain(chainRaw) 

# Format output into an easily readable format 
results = {
    'mu_chi':{'unprocessed':chainRaw[:,:,0].tolist(), 'processed':chainDownsampled[:,0].tolist()},
    'sigma_chi':{'unprocessed':chainRaw[:,:,1].tolist(), 'processed':chainDownsampled[:,1].tolist()},
    'MF_cost':{'unprocessed':chainRaw[:,:,2].tolist(), 'processed':chainDownsampled[:,2].tolist()},
    'sigma_cost':{'unprocessed':chainRaw[:,:,3].tolist(), 'processed':chainDownsampled[:,3].tolist()},
    'cost_min':{'unprocessed':chainRaw[:,:,4].tolist(), 'processed':chainDownsampled[:,4].tolist()},
    'Bq':{'unprocessed':chainRaw[:,:,5].tolist(), 'processed':chainDownsampled[:,5].tolist()}
} 

# Save
savename = froot+"data/{0}injections/{1}.json".format(numinjections,model)
with open(savename, "w") as outfile:
    json.dump(results, outfile)
print(f'Done! Run saved at {savename}')