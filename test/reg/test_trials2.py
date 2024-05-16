#!/usr/bin/python3
"""Template for trial tests

@author: Thilo Bronnenmeyer, kiteswarms 2018

- edit: Rachel Leuthold, Jochem De Schutter ALU-FR 2020-21
"""

import collections
import copy
import logging
import pdb
import casadi.tools as cas

import awebox.opts.kite_data.ampyx_data as ampyx_data
import awebox.opts.kite_data.bubbledancer_data as bubbledancer_data
import awebox.opts.kite_data.boeing747_data as boeing747_data
from ampyx_ap2_settings import set_ampyx_ap2_settings
import awebox.opts.options as options
import awebox.trial as awe_trial
import awebox.tools.save_operations as save_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import numpy as np

from awebox.logger.logger import Logger as awelogger
import matplotlib.pyplot as plt
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
awelogger.logger.setLevel(10)



logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.DEBUG)


# 1
def test_single_kite_basic_health(final_homotopy_step='final'):
    trial_name = 'single_kite_basic_health_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


# 2
def test_single_kite(final_homotopy_step='final'):
    trial_name = 'single_kite_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


# 3
def test_single_kite_6_dof_basic_health(n_k=9, final_homotopy_step='final', cost_factor_power=1., cost_psi_1=1.e3):
    trial_name = 'single_kite_6_dof_basic_health_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step, n_k=n_k, cost_factor_power=cost_factor_power, cost_psi_1=cost_psi_1)
    return None


# 4
def test_single_kite_6_dof(final_homotopy_step='final', cost_factor_power=1., cost_psi_1=1.e3):
    trial_name = 'single_kite_6_dof_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step, cost_factor_power=cost_factor_power, cost_psi_1=cost_psi_1)
    return None


# 5
def test_poly(final_homotopy_step='final'):
    trial_name = 'poly_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


# 6
def test_force_not_stress(final_homotopy_step='final'):
    trial_name = 'force_not_stress_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None

# 7
def test_drag_mode(final_homotopy_step='final'):
    trial_name = 'drag_mode_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


# 8
def test_save_trial():
    trial_name = 'save_trial'
    run_a_solve_and_check_test(trial_name)
    return None


# 9
def test_dual_kite(final_homotopy_step='final', n_k=9, cost_psi_1=1e3, cost_factor_power=1e3):
    trial_name = 'dual_kite_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step, n_k=n_k, cost_psi_1=cost_psi_1, cost_factor_power=cost_factor_power)
    return None


# 10
def test_dual_kite_basic_health(final_homotopy_step='final', n_k=9, cost_psi_1=1e3, cost_factor_power=1e3):
    trial_name = 'dual_kite_basic_health_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step, n_k=n_k, cost_psi_1=cost_psi_1, cost_factor_power=cost_factor_power)
    return None


# 11
def test_dual_kite_6_dof(final_homotopy_step='final'):
    trial_name = 'dual_kite_6_dof_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


# 12
def test_dual_kite_6_dof_basic_health(final_homotopy_step='final'):
    trial_name = 'dual_kite_6_dof_basic_health_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


# 13
def test_dual_kite_tracking():
    trial_name = 'dual_kite_tracking_trial'
    run_a_solve_and_check_test(trial_name)
    return None


# 14
def test_dual_kite_tracking_winch():
    trial_name = 'dual_kite_tracking_winch_trial'
    run_a_solve_and_check_test(trial_name)
    return None


# 15
def test_vortex_force_zero_basic_health(final_homotopy_step='final'):
    trial_name = 'vortex_force_zero_basic_health_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


# 16
def test_vortex_force_zero(final_homotopy_step='final'):
    trial_name = 'vortex_force_zero_trial'
    run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
    return None


# 17
def test_vortex_basic_health(final_homotopy_step='final'):
    options_dict = generate_options_dict()
    trial_name = 'vortex_basic_health_trial'
    solve_trial(options_dict[trial_name], trial_name, final_homotopy_step=final_homotopy_step)
    return None


# 18
def test_vortex(final_homotopy_step='final'):
    options_dict = generate_options_dict()
    trial_name = 'vortex_trial'
    solve_trial(options_dict[trial_name], trial_name, final_homotopy_step=final_homotopy_step)
    return None


# def test_small_dual_kite(final_homotopy_step='final'):
#     trial_name = 'small_dual_kite_trial'
#     run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
#     return None

#
# def test_large_dual_kite(final_homotopy_step='final'):
#     trial_name = 'large_dual_kite_trial'
#     run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
#     return None


# def test_actuator_qaxi(final_homotopy_step='final'):
#     trial_name = 'actuator_qaxi_trial'
#     run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
#     return None
#
# def test_actuator_qaxi_basic_health(final_homotopy_step='final'):
#     trial_name = 'actuator_qaxi_basic_health_trial'
#     run_a_solve_and_check_test(trial_name, final_homotopy_step=final_homotopy_step)
#     return None
#
#
# def test_actuator_uaxi():
#     trial_name = 'actuator_uaxi_trial'
#     run_a_solve_and_check_test(trial_name)
#     return None
#
#
# def test_actuator_qasym():
#     trial_name = 'actuator_qasym_trial'
#     run_a_solve_and_check_test(trial_name)
#     return None
#
#
# def test_actuator_uasym():
#     trial_name = 'actuator_uasym_trial'
#     run_a_solve_and_check_test(trial_name)
#     return None
#
#
# def test_actuator_comparison():
#     trial_name = 'actuator_comparison_trial'
#     run_a_solve_and_check_test(trial_name)
#     return None




def make_basic_health_variant(base_options):
    basic_health_options = copy.deepcopy(base_options)

    basic_health_options['user_options.trajectory.lift_mode.windings'] = 1
    basic_health_options['nlp.n_k'] = 10 #12  # try to decrease this.
    basic_health_options['nlp.collocation.d'] = 3
    basic_health_options['nlp.collocation.u_param'] = 'zoh'
    basic_health_options['solver.hippo_strategy'] = False

    basic_health_options['solver.health_check.when'] = 'always'
    basic_health_options['solver.homotopy_method.advance_despite_max_iter'] = False
    basic_health_options['solver.homotopy_method.advance_despite_ill_health'] = False
    basic_health_options['solver.initialization.check_reference'] = True
    basic_health_options['solver.initialization.check_feasibility.raise_exception'] = True
    basic_health_options['solver.max_iter'] = 300
    basic_health_options['solver.ipopt.autoscale'] = False
    basic_health_options['solver.health_check.raise_exception'] = True
    basic_health_options['solver.health_check.spy_matrices'] = False
    basic_health_options['nlp.collocation.name_constraints'] = True
    basic_health_options['solver.health_check.help_with_debugging'] = True
    basic_health_options['quality.when'] = 'never'
    basic_health_options['visualization.cosmetics.variables.si_or_scaled'] = 'scaled'
    basic_health_options['solver.health_check.save_health_indicators'] = True

    return basic_health_options


def generate_options_dict(n_k=9, cost_factor_power=1e5, cost_psi_1=1e3):
    """
    Set options for the trials that should be tested and store them in dictionary
    :return: dictionary with trial options
    """

    # i think these tests are intended to test whether it's possible to use the awebox
    # 'straight out of the box', so i think we should try to keep the test options as
    # close to default as possible.

    # set options
    single_kite_options = {}
    single_kite_options['user_options.system_model.architecture'] = {1: 0}
    single_kite_options['user_options.system_model.kite_dof'] = 3
    single_kite_options['user_options.kite_standard'] = ampyx_data.data_dict()
    single_kite_options['user_options.trajectory.system_type'] = 'lift_mode'
    single_kite_options['user_options.trajectory.lift_mode.windings'] = 1
    single_kite_options['user_options.trajectory.fixed_params'] = {'diam_t': 2e-3}
    single_kite_options['user_options.induction_model'] = 'not_in_use'  # don't include induction effects

    single_kite_options['user_options.kite_standard.aero_validity.beta_max_deg'] = 20.
    single_kite_options['user_options.kite_standard.aero_validity.beta_min_deg'] = -20.
    single_kite_options['user_options.kite_standard.aero_validity.alpha_max_deg'] = 9.0
    single_kite_options['user_options.kite_standard.aero_validity.alpha_min_deg'] = -6.0
    single_kite_options['user_options.kite_standard.geometry.delta_max'] = np.array([20., 30., 30.]) * np.pi / 180.
    single_kite_options['user_options.kite_standard.geometry.ddelta_max'] = np.array([2., 2., 2.])

    single_kite_options['model.tether.control_var'] = 'ddl_t'  # tether acceleration control

    # single_kite_options['solver.initialization.groundspeed'] = 15.
    # single_kite_options['solver.initialization.inclination_deg'] = 45.
    # single_kite_options['solver.initialization.cone_deg'] = 15.
    # single_kite_options['solver.initialization.l_t'] = 200.

    single_kite_options['model.model_bounds.airspeed.include'] = True
    single_kite_options['params.model_bounds.airspeed_limits'] = np.array([10, 32.0])  # [m/s]

    single_kite_options['model.system_bounds.x.l_t'] = [10.0, 700.0]  # [m]
    single_kite_options['model.system_bounds.x.dl_t'] = [-15.0, 20.0]  # [m/s]
    single_kite_options['model.system_bounds.x.ddl_t'] = [-2.4, 2.4]  # [m/s^2]
    single_kite_options['model.system_bounds.x.q'] = [np.array([-cas.inf, -cas.inf, 100.0]), np.array([cas.inf, cas.inf, cas.inf])]
    single_kite_options['model.system_bounds.x.omega'] = [np.array(3*[-50.0*np.pi/180.0]), np.array(3*[50.0*np.pi/180.0])]
    single_kite_options['model.system_bounds.theta.t_f'] = [5., 70.]  # [s]

    single_kite_options['nlp.collocation.u_param'] = 'zoh'
    single_kite_options['nlp.n_k'] = 30
    single_kite_options['solver.cost_factor.power'] = 1e6
    single_kite_options['solver.linear_solver'] = 'ma57'
    single_kite_options['solver.homotopy_method.advance_despite_max_iter'] = False
    single_kite_options['solver.initialization.check_reference'] = True  # check that the specified initial guess is not infeasible

    single_kite_options['visualization.cosmetics.plot_bounds'] = True
    single_kite_options['visualization.cosmetics.trajectory.kite_bodies'] = True
    single_kite_options['visualization.cosmetics.trajectory.kite_aero_dcm'] = False
    single_kite_options['visualization.cosmetics.outputs.include_solution'] = True

    # single_kite_options['solver.cost.fictitious.2'] = 1e0
    # single_kite_options['solver.cost.tracking.1'] = 1e-2

    single_kite_basic_health_options = make_basic_health_variant(single_kite_options)

    single_kite_6_dof_options = copy.deepcopy(single_kite_options)
    single_kite_6_dof_options['user_options.system_model.kite_dof'] = 6
    single_kite_6_dof_options['solver.cost_factor.power'] = 1e5

    single_kite_6_dof_basic_health_options = make_basic_health_variant(single_kite_6_dof_options)

    poly_options = copy.deepcopy(single_kite_options)
    poly_options['nlp.collocation.u_param'] = 'poly'
    poly_options['model.system_bounds.theta.t_f'] = [5., 30.]  # [s]

    force_not_stress_options = copy.deepcopy(single_kite_options)
    force_not_stress_options['model.model_bounds.tether_stress.include'] = False
    force_not_stress_options['model.model_bounds.tether_force.include'] = True
    force_not_stress_options['params.model_bounds.tether_force_limits'] = np.array([50, 1800.0])

    drag_mode_options = copy.deepcopy(single_kite_options)
    drag_mode_options['user_options.trajectory.system_type'] = 'drag_mode'
    drag_mode_options['quality.test_param.power_balance_thresh'] = 2.
    drag_mode_options['model.system_bounds.theta.t_f'] = [20., 70.]  # [s]

    save_trial_options = copy.deepcopy(single_kite_options)
    save_trial_options['solver.save_trial'] = True

    dual_kite_options = copy.deepcopy(single_kite_options)
    dual_kite_options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1}
    dual_kite_options['solver.initialization.cone_deg'] = 30.
    dual_kite_options['model.system_bounds.theta.t_f'] = [5., 20.]  # [s]
    dual_kite_options['solver.cost_factor.power'] = 1e6 #5

    dual_kite_basic_health_options = make_basic_health_variant(dual_kite_options)

    dual_kite_6_dof_options = copy.deepcopy(dual_kite_options)
    dual_kite_6_dof_options['user_options.system_model.kite_dof'] = 6
    dual_kite_6_dof_options['model.system_bounds.theta.t_f'] = [5., 70.]  # [s]

    dual_kite_6_dof_basic_health_options = make_basic_health_variant(dual_kite_6_dof_options)

    small_dual_kite_options = copy.deepcopy(dual_kite_6_dof_options)
    small_dual_kite_options['user_options.kite_standard'] = bubbledancer_data.data_dict()
    small_dual_kite_options['user_options.trajectory.lift_mode.windings'] = 1
    small_dual_kite_options['solver.cost_factor.power'] = 1e7
    # small_dual_kite_options['model.system_bounds.theta.t_f'] = [2., 60.]

    large_dual_kite_options = copy.deepcopy(dual_kite_6_dof_options)
    large_dual_kite_options['user_options.kite_standard'] = boeing747_data.data_dict()
    large_dual_kite_options['user_options.trajectory.lift_mode.windings'] = 1
    large_dual_kite_options['solver.initialization.theta.l_s'] = 60. * 10.
    large_dual_kite_options['solver.initialization.l_t'] = 2.e3
    large_dual_kite_options['model.system_bounds.theta.t_f'] = [1.e-3, 500.]
    large_dual_kite_options['model.model_bounds.tether_force.include'] = False
    large_dual_kite_options['model.model_bounds.tether_stress.include'] = True
    large_dual_kite_options['solver.cost.tracking.0'] = 1e-1
    large_dual_kite_options['model.system_bounds.theta.t_f'] = [5., 120.]

    actuator_qaxi_options = copy.deepcopy(dual_kite_6_dof_options)
    actuator_qaxi_options['user_options.kite_standard'] = ampyx_data.data_dict()
    actuator_qaxi_options['user_options.induction_model'] = 'actuator'
    actuator_qaxi_options['model.aero.actuator.steadyness'] = 'quasi-steady'
    actuator_qaxi_options['model.aero.actuator.symmetry'] = 'axisymmetric'
    actuator_qaxi_options['visualization.cosmetics.trajectory.actuator'] = True
    actuator_qaxi_options['visualization.cosmetics.trajectory.kite_bodies'] = True
    actuator_qaxi_options['model.system_bounds.theta.a'] = [-0., 0.5]
    # # actuator_qaxi_options['model.aero.actuator.normal_vector_model'] = 'least_squares'
    #
    # # actuator_qaxi_options['solver.cost.gamma.1'] = 1.e2  # 1e3 fictitious problem by 1e-1
    # # actuator_qaxi_options['solver.cost.psi.1'] = 1.e2  # 1e4 power problem scaled by 1e-2
    # # # actuator_qaxi_options['solver.cost.theta_regularisation.0'] = 1.e0
    # # actuator_qaxi_options['solver.cost.iota.1'] = 1.e2  # 1e3 induction problem scaled by 1e-1
    # # # actuator_qaxi_options['solver.max_iter'] = 90
    # # # actuator_qaxi_options['solver.max_iter_hippo'] = 90
    # # actuator_qaxi_options['solver.cost_factor.power'] = 1e5  # 1e4 -> high reg in final step
    #
    # # actuator_qaxi_options['solver.cost.theta_regularisation.0'] = 1.e-1
    # actuator_qaxi_options['user_options.trajectory.lift_mode.windings'] = 3
    # # actuator_qaxi_options['solver.cost.beta.0'] = 1.e0
    # # actuator_qaxi_options['solver.weights.r'] = 1e-1
    # # actuator_qaxi_options['solver.weights.q'] = 1e2
    # # actuator_qaxi_options['solver.weights.dq'] = 1e2
    # # actuator_qaxi_options['solver.cost.u_regularisation.0'] = 1e-5

    actuator_qaxi_basic_health_options = make_basic_health_variant(actuator_qaxi_options)

    actuator_uaxi_options = copy.deepcopy(actuator_qaxi_options)
    actuator_uaxi_options['model.aero.actuator.steadyness'] = 'unsteady'
    actuator_uaxi_options['model.model_bounds.tether_stress.scaling'] = 10.

    actuator_qasym_options = copy.deepcopy(actuator_qaxi_options)
    actuator_qasym_options['model.aero.actuator.symmetry'] = 'asymmetric'
    actuator_qasym_options['solver.cost.psi.1'] = 1.e1

    actuator_uasym_options = copy.deepcopy(actuator_qaxi_options)
    actuator_uasym_options['model.aero.actuator.symmetry'] = 'asymmetric'
    actuator_uasym_options['model.aero.actuator.symmetry'] = 'asymmetric'
    actuator_uasym_options['solver.cost.psi.1'] = 1.e1

    actuator_comparison_options = copy.deepcopy(actuator_qaxi_options)
    actuator_comparison_options['model.aero.actuator.steadyness_comparison'] = ['q', 'u']
    actuator_comparison_options['user_options.system_model.kite_dof'] = 6

    vortex_options = copy.deepcopy(single_kite_6_dof_options)
    vortex_options['user_options.induction_model'] = 'vortex'
    vortex_options['quality.test_param.vortex_truncation_error_thresh'] = 1e20
    vortex_options['visualization.cosmetics.trajectory.wake_nodes'] = True
    vortex_options['model.aero.vortex.wake_nodes'] = 2
    vortex_options['solver.cost_factor.power'] = 1e4
    vortex_options['model.aero.vortex.degree_of_induced_velocity_lifting'] = 1
    vortex_options['model.aero.vortex.rate_of_change_scaling_factor'] = 1.e-3
    vortex_options['model.scaling.other.position_scaling_method'] = 'altitude'
    # vortex_options['model.system_bounds.theta.t_f'] = [50., 70.]  # [s]

    vortex_basic_health_options = make_basic_health_variant(vortex_options)

    vortex_force_zero_options = copy.deepcopy(vortex_options)
    vortex_force_zero_options['model.aero.induction.force_zero'] = True
    # vortex_force_zero_options['nlp.collocation.d'] = 4

    vortex_force_zero_basic_health_options = make_basic_health_variant(vortex_force_zero_options)
    vortex_force_zero_basic_health_options['nlp.n_k'] = 13

    dual_kite_tracking_options = copy.deepcopy(dual_kite_6_dof_options)
    dual_kite_tracking_options['user_options.trajectory.type'] = 'tracking'
    dual_kite_tracking_options['user_options.trajectory.lift_mode.windings'] = 1

    dual_kite_tracking_winch_options = copy.deepcopy(dual_kite_tracking_options)
    dual_kite_tracking_winch_options['user_options.trajectory.tracking.fix_tether_length'] = False

    # nominal landing
    nominal_landing_options = copy.deepcopy(dual_kite_options)
    nominal_landing_options['user_options.trajectory.type'] = 'nominal_landing'
    nominal_landing_options['user_options.trajectory.transition.initial_trajectory'] = 'dual_kite_trial.dict'
    nominal_landing_options['solver.initialization.initialization_type'] = 'modular'

    # compromised landing
    compromised_landing_options = copy.deepcopy(nominal_landing_options)
    compromised_landing_options['user_options.trajectory.type'] = 'compromised_landing'
    compromised_landing_options['model.model_bounds.dcoeff_compromised_factor'] = 0.0
    compromised_landing_options['user_options.trajectory.compromised_landing.emergency_scenario'] = ('broken_roll', 2)
    compromised_landing_options['user_options.trajectory.compromised_landing.xi_0_initial'] = 0.8

    # define options list
    options_dict = collections.OrderedDict()
    options_dict['single_kite_trial'] = single_kite_options
    options_dict['single_kite_basic_health_trial'] = single_kite_basic_health_options
    options_dict['single_kite_6_dof_trial'] = single_kite_6_dof_options
    options_dict['single_kite_6_dof_basic_health_trial'] = single_kite_6_dof_basic_health_options
    options_dict['poly_trial'] = poly_options
    options_dict['force_not_stress_trial'] = force_not_stress_options
    options_dict['drag_mode_trial'] = drag_mode_options
    options_dict['save_trial'] = save_trial_options
    options_dict['dual_kite_trial'] = dual_kite_options
    options_dict['dual_kite_basic_health_trial'] = dual_kite_basic_health_options
    options_dict['small_dual_kite_trial'] = small_dual_kite_options
    options_dict['large_dual_kite_trial'] = large_dual_kite_options
    options_dict['dual_kite_6_dof_trial'] = dual_kite_6_dof_options
    options_dict['dual_kite_6_dof_basic_health_trial'] = dual_kite_6_dof_basic_health_options
    options_dict['actuator_qaxi_trial'] = actuator_qaxi_options
    options_dict['actuator_qaxi_basic_health_trial'] = actuator_qaxi_basic_health_options
    options_dict['actuator_uaxi_trial'] = actuator_uaxi_options
    options_dict['actuator_qasym_trial'] = actuator_qasym_options
    options_dict['actuator_uasym_trial'] = actuator_uasym_options
    options_dict['actuator_comparison_trial'] = actuator_comparison_options
    options_dict['vortex_force_zero_trial'] = vortex_force_zero_options
    options_dict['vortex_force_zero_basic_health_trial'] = vortex_force_zero_basic_health_options
    options_dict['vortex_trial'] = vortex_options
    options_dict['vortex_basic_health_trial'] = vortex_basic_health_options
    options_dict['dual_kite_tracking_trial'] = dual_kite_tracking_options
    options_dict['dual_kite_tracking_winch_trial'] = dual_kite_tracking_winch_options
    # options_dict['nominal_landing_trial'] = nominal_landing_options
    # options_dict['compromised_landing_trial'] = compromised_landing_options

    return options_dict


def run_a_solve_and_check_test(trial_name, final_homotopy_step='final', n_k=9, cost_factor_power=1., cost_psi_1=1.e3):
    """
    Solve one individual trial and run tests on it
    :param trial_name: name of the trial
    :return: None
    """

    options_dict = generate_options_dict(n_k=n_k, cost_factor_power=cost_factor_power, cost_psi_1=cost_psi_1)
    trial_options = options_dict[trial_name]

    # compute trajectory solution
    trial = solve_trial(trial_options, trial_name, final_homotopy_step=final_homotopy_step)

    # evaluate results
    if hasattr(trial, 'quality') and hasattr(trial.quality, 'results'):
        evaluate_results(trial.quality.results, trial_name)

    if not trial.optimization.solve_succeeded:
        message = 'optimization of trial ' + trial_name + ' failed'
        raise Exception(message)

    return None


def evaluate_results(results, trial_name):
    for test_name in list(results.keys()):
        assert results[test_name], 'Test failed for ' + trial_name + ', Test regarding ' + test_name + ' failed.'

    return None


def solve_trial(trial_options, trial_name, final_homotopy_step='final'):
    """
    Set up and solve trial
    :return: solved trial
    """

    trial = awe_trial.Trial(trial_options, trial_name)
    trial.build()
    trial.optimize(final_homotopy_step=final_homotopy_step)

    return trial

# #
# for n_k in range(9, 14):
#     for cost_psi_1 in [1e3, 1e2, 1e4]:
#         for cost_factor_power in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
#             table_dict = {'n_k': n_k, 'cost_psi_1': cost_psi_1, 'cost_factor_power': cost_factor_power, 'pass': False}
#             filename = 'test_options'
#             try:
#                 test_dual_kite_basic_health(final_homotopy_step='final', n_k=n_k, cost_psi_1=cost_psi_1,
#                                             cost_factor_power=cost_factor_power)
#                 test_dual_kite(final_homotopy_step='final', n_k=n_k, cost_psi_1=cost_psi_1,
#                                             cost_factor_power=cost_factor_power)
#                 # test_single_kite_6_dof_basic_health(cost_factor_power=cost_factor_power)
#                 # test_single_kite_6_dof(cost_factor_power=cost_factor_power)
#                 table_dict['pass'] = True
#             except:
#                 pass
#             save_op.write_or_append_two_column_dict_to_csv(table_dict, filename)


# test_single_kite_basic_health()
# test_single_kite()
# test_single_kite_6_dof_basic_health()
# test_single_kite_6_dof()
# test_poly()
# test_force_not_stress()
# test_drag_mode()
# test_save_trial()
# test_dual_kite_basic_health()
# test_dual_kite()
# test_dual_kite_6_dof_basic_health()
# test_dual_kite_6_dof()
# test_dual_kite_tracking()
# test_dual_kite_tracking_winch()
test_vortex_force_zero_basic_health() #<<
test_vortex_force_zero()
test_vortex_basic_health()
test_vortex()
# # # # test_small_dual_kite() #<< this does not work. "Test failed for small_dual_kite_trial, Test regarding power_dominance failed."
# # # # test_large_dual_kite() << ?
# # # # test_actuator_qaxi_basic_health() #final_homotopy_step='induction')
# # # # test_actuator_qaxi()
# # # # test_actuator_qasym()
# # # # test_actuator_uaxi()
# # # # test_actuator_uasym()
# # # # test_actuator_comparison()
