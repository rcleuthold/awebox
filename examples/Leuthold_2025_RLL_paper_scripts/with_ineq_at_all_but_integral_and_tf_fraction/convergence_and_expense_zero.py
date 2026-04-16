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


def run(inputs={}):

    n_k = inputs['n_k']
    periods_tracked = inputs['periods_tracked']
    tol = inputs['tol']
    mu_hippo = inputs['mu_hippo']
    solver = inputs['solver']

    base_name = 'convergence'
    wake_nodes = int(np.ceil(n_k * periods_tracked + 1))

    # basic options
    options = {}
    options = help_op.get_basic_options_for_convergence_expense_and_comparison(options)
    
    options['model.scaling.other.flight_radius_estimate'] = inputs['flight_radius_estimate'] 
    options['model.scaling.other.period_estimate'] = inputs['period_estimate']
    options['model.scaling.other.position_scaling_method'] = inputs['position_scaling_method']
    options['model.scaling.other.force_scaling_method'] = inputs['force_scaling_method']
    options['model.scaling.other.tension_estimate'] = inputs['tension_estimate']
    options['model.scaling.other.power_estimate'] = inputs['power_estimate']

    # allow a reduction of the problem for testing purposed
    options['nlp.n_k'] = n_k
    options['model.aero.vortex.wake_nodes'] = wake_nodes

    options['solver.linear_solver'] = solver
    options['solver.tol'] = tol
    options['solver.mu_hippo'] = mu_hippo
    if mu_hippo == False:
        options['solver.mu_hippo'] = 1e-2
        options['solver.hippo_strategy'] = False

    options['user_options.induction_model'] = 'not_in_use'

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
    options['model.aero.actuator.normal_vector_model'] = 'xhat'
    options['visualization.cosmetics.temporal_epigraph_locations'] = ['switch', 1.0] 

    # use these options only in final comparison. the convergence plots were made with the xhat normal_vector. the only difference has to do with which projections are used in plotting (and in which projections of the induced velocity to store)
    options['visualization.cosmetics.temporal_epigraph_locations'] = [0.32, 'switch', 1.0]
    options['model.aero.actuator.normal_vector_model'] = 'dual'
    
    
    
    
    
    
    
    ##options['user_options.trajectory.lift_mode.windings'] = 1
    ##options['nlp.n_k'] = 7 # try to decrease this.
    ##options['nlp.collocation.d'] = 2
    #options['nlp.collocation.u_param'] = 'zoh'
    #options['solver.hippo_strategy'] = False

    #options['solver.health_check.when'] = 'success'
    #options['nlp.collocation.name_constraints'] = True
    #options['solver.health_check.help_with_debugging'] = False
    #3options['model.scaling.other.print_help_with_scaling'] = True

    #options['solver.homotopy_method.advance_despite_max_iter'] = False
    #options['solver.homotopy_method.advance_despite_ill_health'] = False
    #options['solver.homotopy_method.consider_restoration_as_failure'] = False #True
    #options['solver.health_check.raise_exception'] = False #True
    #options['solver.initialization.check_reference'] = True
    #options['solver.initialization.check_feasibility.raise_exception'] = False #True
    ##options['solver.max_iter'] = 500
    #options['solver.ipopt.autoscale'] = False
    #options['solver.health_check.spy_matrices'] = False
    #options['quality.when'] = 'never'
    #options['visualization.cosmetics.variables.si_or_scaled'] = 'si'
    #options['solver.health_check.save_health_indicators'] = True
    #options['solver.health_check.thresh.condition_number'] = 1e10
    #options['solver.tol'] = 1e-8

    
    
    
    
    
    
    
    
    
    
    
    # build trial and optimize
    trial_name_vortex = help_op.build_unique_trial_name(base_name, inputs)
    trial_vortex = awe_trial.Trial(options, trial_name_vortex)
    trial_vortex.build()

    trial_vortex.optimize(final_homotopy_step='final')

    #import os
    #os.environ["OMP_NUM_THREADS"] = "1"

    trial_vortex.print_cost_information()
    help_op.save_results_including_figures(trial_vortex, options)

    return None

    
def call_by_pt(n_k, pt, ipopt_tol=1e-8, pt_min=1e-3, mu_hippo=1e-1, use_hippo_strategy=True, solver='ma86', flight_radius_estimate='synthesized', period_estimate='synthesized', position_scaling_method='radius', force_scaling_method='synthesized', tension_estimate='synthesized', power_estimate='synthesized'):
    import gc
    from glob import glob
    #if pt > pt_min:
    inputs = {}
    inputs['n_k'] = n_k
    inputs['periods_tracked'] = pt
    inputs['tol'] = ipopt_tol
    inputs['mu_hippo'] = mu_hippo
    if not use_hippo_strategy:
        inputs['mu_hippo'] = False
    inputs['solver'] = solver
    
    inputs['flight_radius_estimate'] = flight_radius_estimate
    inputs['period_estimate'] = period_estimate
    inputs['position_scaling_method'] = position_scaling_method
    inputs['force_scaling_method'] = force_scaling_method
    inputs['tension_estimate'] = tension_estimate
    inputs['power_estimate'] = power_estimate

    trial_name = ''
    for name, val in inputs.items():
        trial_name += '_' + name + '_' + str(val)
    #if not glob('*' + trial_name + '*'):
    trial = run(inputs)
    del trial
    gc.collect()
    return None


def call_by_memory(n_k, memory_gb, ipopt_tol=1e-8, pt_min=1e-3, mu_hippo=1e-1):

    # curve fit for memory [GB]: 3.56715 + 0.00121953 V
    aa = 5.79095
    bb = 0.0010823
    def estimate_periods_tracked(aa, bb, n_k, d, kites, mem_gb):
        p1 = -aa - 3. * bb * (1. + n_k + d * n_k) * kites + mem_gb
        p2 = 3. * bb * n_k * (1. + n_k + d * n_k) * kites
        pt = p1 / p2
        return pt

    collocation_d = 4

    pt = estimate_periods_tracked(aa, bb, n_k, collocation_d, 2, memory_gb)
    call_by_pt(n_k, pt, ipopt_tol=ipopt_tol, pt_min=pt_min, mu_hippo=mu_hippo)
    return None
    

if __name__ == "__main__":
    n_k = 25
    pt = 2
    call_by_pt(n_k, p_t)
