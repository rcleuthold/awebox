#!/usr/bin/python3
from platform import architecture

import matplotlib
# matplotlib.use('TkAgg')

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

    base_name = 'convergence'
    wake_nodes = int(np.ceil(n_k * periods_tracked + 1))

    # basic options
    options = {}
    options = help_op.get_basic_options_for_convergence_expense_and_comparison(options)

    # allow a reduction of the problem for testing purposed
    options['nlp.n_k'] = n_k
    options['model.aero.vortex.wake_nodes'] = wake_nodes

    options['solver.tol'] = tol

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
    
    
    # build trial and optimize
    trial_name_vortex = help_op.build_unique_trial_name(base_name, inputs)
    trial_vortex = awe_trial.Trial(options, trial_name_vortex)
    trial_vortex.build()

    trial_vortex.optimize(final_homotopy_step='final')

    trial_vortex.print_cost_information()
    help_op.save_results_including_figures(trial_vortex, options)

    return None


if __name__ == "__main__":

    tol = 1e-8
    pt_min = 1e-3

    #n_k = 15
    #pt = 1.5 #.75
    #
    #inputs = {}
    #inputs['n_k'] = n_k
    #inputs['periods_tracked'] = pt
    #inputs['tol'] = tol
    #trial = run(inputs)

    import gc
    from glob import glob

    # curve fit for memory [GB]: 3.56715 + 0.00121953 V
    aa = 5.79095
    bb = 0.0010823
    # # run: 0.3 * [20]
    # 0.8 * [30]


    def estimate_periods_tracked(aa, bb, n_k, d, kites, mem_gb):
         p1 = -aa - 3. * bb * (1. + n_k + d * n_k) * kites + mem_gb
         p2 = 3. * bb * n_k * (1. + n_k + d * n_k) * kites
         pt = p1/p2
         return pt

    total_memory_gb = 128
    target_memory_gb = 0.60 * total_memory_gb
    collocation_d = 4

    for n_k in [50]: #[40, 50, 20, 25, 35, 60, 45, 15, 55]: #30
        pt = estimate_periods_tracked(aa, bb, n_k, collocation_d, 2, target_memory_gb)
        if pt > pt_min:
         inputs = {}
         inputs['n_k'] = n_k
         inputs['periods_tracked'] = pt
         inputs['tol'] = tol

         trial_name = ''
         for name, val in inputs.items():
             trial_name += '_' + name + '_' + str(val)
         if not glob('*' + trial_name + '*'):
             trial = run(inputs)
             del trial
         gc.collect()

    if save_op.running_on_aws_ec2():
        save_op.stop_this_aws_ec2_instance()
