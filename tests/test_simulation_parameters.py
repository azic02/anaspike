import copy

import nest



simulation = {
    'duration': 5000.
    }

spatial = {
    'extent': {
        'x': 2.,
        'y': 2.
        }
    }

__shared_population_params = {
    'neuronal': {
        'params': {
            'C_m': 1.0,                        # membrane capacity (pF)
            'E_L': 0.,                         # resting membrane potential (mV)
            'I_e': 0.,                         # external input current (pA)
            'V_m': 0.,                         # membrane potential (mV)
            'V_reset': 10.,                    # reset membrane potential after a spike (mV)
            'V_th': 20.,                       # spike threshold (mV)
            't_ref': 2.0,                      # refractory period (ms)
            'tau_m': 20.,                      # membrane time constant (ms)
            },
        'positions': None
        },
    'synaptic': {
        'conn_spec': {
            'rule': 'pairwise_bernoulli',
            'p': 1.0,
            'mask': {'rectangular': {'lower_left': [-0.5, -0.5], 'upper_right': [0.5, 0.5]}},
            'allow_autapses': True
            },
        'syn_spec': {
            'synapse_model': 'static_synapse',
            'delay': 1.5,                      # synaptic transmission delay (ms)
            }
        }
    }

populations = {
   'exc': copy.deepcopy(__shared_population_params),
   'inh': copy.deepcopy(__shared_population_params)
   }

populations['exc']['synaptic']['syn_spec']['weight'] = 1.0         # synaptic weight (mV)
populations['exc']['neuronal']['positions'] = nest.spatial.grid(
    shape=[20, 20],
    extent=[spatial['extent']['x'], spatial['extent']['y']],
    edge_wrap=True
    )

populations['inh']['synaptic']['syn_spec']['weight'] = -5. * populations['exc']['synaptic']['syn_spec']['weight']
populations['inh']['neuronal']['positions'] = nest.spatial.grid(
    shape=[10, 10],
    extent=[spatial['extent']['x'], spatial['extent']['y']],
    edge_wrap=True
    )


V_th = __shared_population_params['neuronal']['params']['V_th']
tau_m = __shared_population_params['neuronal']['params']['tau_m']
exc_weight = populations['exc']['synaptic']['syn_spec']['weight']
nu_th = V_th / (exc_weight * tau_m)                     # external rate needed to evoke activity (spikes/ms)
nu_ex = 2.0 * nu_th                                     # set external rate above threshold
poisson_generator = {
    'params': {
        'rate': 1e3 * nu_ex                             # external rate (spikes/s)
        },
    'syn_spec': populations['exc']['synaptic']['syn_spec']
    }

