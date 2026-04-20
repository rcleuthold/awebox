#!/usr/bin/python3
from platform import architecture

import matplotlib
matplotlib.use("Agg")   # ← MUST be here, before pyplot

import awebox as awe
import matplotlib.pyplot as plt

import pickle
import numpy as np
import csv
import os

from datetime import date
import random


import awebox.trial as awe_trial
import awebox.opts.kite_data.ampyx_data as ampyx_data
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.save_operations as save_op

import awebox.viz.wake as wake_viz
import awebox.opti.initialization_dir.initialization as initialization


import helpful_operations as help_op

from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas

awelogger.logger.setLevel(10)


base_name = 'convergence'

def run(inputs={}):

    # basic options
    options = {}
    options = help_op.get_basic_options_for_convergence_expense_and_comparison(options)

    # allow a reduction of the problem for testing purposed
    if 'nlp.n_k' in inputs.keys():
        n_k = inputs['nlp.n_k']
    else:
        n_k = options['nlp.n_k']
    periods_tracked = inputs['periods_tracked']
    wake_nodes = int(np.ceil(n_k * periods_tracked + 1))
    options['model.aero.vortex.wake_nodes'] = wake_nodes

    if ('solver.mu_hippo' in inputs.keys()) and (inputs['solver.mu_hippo'] == False):
        inputs['solver.mu_hippo'] = 1e-2
        inputs['solver.hippo_strategy'] = False

    for name, val in inputs.items():
        if name != 'periods_tracked':
            options[name] = inputs[name]


    # visualization
    options['visualization.cosmetics.save_figs'] = True
    options['visualization.cosmetics.save.format_list'] = ['pdf']
    options['visualization.cosmetics.animation.snapshot_index'] = -1
    options['visualization.cosmetics.trajectory.body_cross_sections_per_meter'] = 10 / options['user_options.kite_standard']['geometry']['b_ref']
    options['visualization.cosmetics.trajectory.wake_nodes'] = True
    options['visualization.cosmetics.trajectory.kite_aero_dcm'] = True
    options['visualization.cosmetics.trajectory.trajectory_rotation_dcm'] = True
    options['visualization.cosmetics.variables.si_or_scaled'] = 'si'
    options['visualization.cosmetics.trajectory.kite_bodies'] = True
    options['visualization.cosmetics.trajectory.reel_in_linestyle'] = '--'  
    options['visualization.cosmetics.trajectory.temporal_epigraph_length_to_span'] = 5.
    
    options['model.aero.vortex.induction_factor_normalizing_speed'] = 'u_ref'  
    options['model.aero.actuator.normal_vector_model'] = 'dual'
    options['visualization.cosmetics.temporal_epigraph_locations'] = [0.32, 'switch', 1.0] 
        
    options['model.scaling.other.print_help_with_scaling'] = True
    
    
    # build trial and optimize
    trial_name_vortex = help_op.build_unique_trial_name(base_name, inputs)
    trial_vortex = awe_trial.Trial(options, trial_name_vortex)
    trial_vortex.build()

    trial_vortex.optimize(final_homotopy_step='final')

    import os
    os.environ["OMP_NUM_THREADS"] = "1"


    trial_vortex.print_cost_information()
    help_op.save_results_including_figures(trial_vortex, options)

    return None

    
def call_by_pt(n_k, pt, inputs):
    import gc
    from glob import glob
    if pt > 0.:
        inputs['nlp.n_k'] = n_k
        inputs['periods_tracked'] = pt

        trail_name = helpful_op.build_unique_trial_name(base_name, inputs)

        if not glob('*' + trial_name + '*'):
            trial = run(inputs)
            del trial
        gc.collect()
    return None


def call_by_memory(n_k, memory_gb, inputs):

    if 'nlp.collocation.d' in inputs.keys:
        collocation_d = inputs['nlp.collocation_d']
    else:
        collocation_d = 4
    
    # curve fit for memory [GB]: 3.56715 + 0.00121953 V
    aa = 5.79095
    bb = 0.0010823
    def estimate_periods_tracked(aa, bb, n_k, d, kites, mem_gb):
        p1 = -aa - 3. * bb * (1. + n_k + d * n_k) * kites + mem_gb
        p2 = 3. * bb * n_k * (1. + n_k + d * n_k) * kites
        pt = p1 / p2
        return pt

    pt = estimate_periods_tracked(aa, bb, n_k, collocation_d, 2, memory_gb)
    call_by_pt(n_k, pt, inputs)
    return None
    

if __name__ == "__main__":
    total_memory_gb = 128
    target_memory_gb = 0.75 * total_memory_gb
    n_k = 25
    call_by_memory(n_k, target_memory_gb)
