import numpy as np
import math
import time

class Auxiliary(object):
    def simulate_step(self, dt, state, control, configs, noise={'nx':0.01,'ny':0.01,'ntheta':0.01,'nvt':0.05,'nvr':0.05}):
        sim_state = {}
        deviation = 0.05

        sim_state['vt'] = max(configs['min_vt'],min(configs['max_vt'], state['vt'] + control['A']*dt + np.random.normal(scale=noise['nvt'], loc=0.0)*dt))
        sim_state['vr'] = max(configs['min_vr'],min(configs['max_vr'], state['vr'] + control['Phi']*dt + np.random.normal(scale=noise['nvr'], loc=0.0)*dt))
        sim_state['x']  = state['x']  + sim_state['vt']*math.cos(state['theta'])*dt + np.random.normal(scale=noise['nx'], loc=0.0)*dt
        sim_state['y']  = state['y']  + sim_state['vt']*math.sin(state['theta'])*dt + np.random.normal(scale=noise['ny'], loc=0.0)*dt
        sim_state['theta'] = state['theta'] + sim_state['vr']*dt + np.random.normal(scale=noise['ntheta'], loc=0.0)*dt
        return sim_state
