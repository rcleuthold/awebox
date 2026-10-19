#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2020 Thilo Bronnenmeyer, Kiteswarms Ltd.
#    Copyright (C) 2016      Elena Malz, Sebastien Gros, Chalmers UT.
#
#    awebox is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    awebox is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with awebox; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
'''
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-21
- edit: jochem de schutter, alu-fr 2019
'''
import pdb

import casadi as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom
import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.actuator_dir.coeff as actuator_coeff
import awebox.mdl.aero.induction_dir.actuator_dir.force as actuator_force
import awebox.mdl.aero.induction_dir.actuator_dir.system as actuator_system

import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools
import awebox.mdl.aero.geometry_dir.geometry as geom

import awebox.tools.constraint_operations as cstr_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op


def model_is_included_in_comparison(options):
    comparison_labels = general_tools.get_option_from_possible_dicts(options, 'comparison_labels', 'actuator')
    any_vor = any(label[:3] == 'act' for label in comparison_labels)
    return any_vor


def get_model_constraints(model_options, atmos, wind, system_variables, parameters, outputs, architecture, scaling):

    variables_si = system_variables['SI']
    cstr_list = cstr_op.ConstraintList()

    support_cstr = get_all_model_support_constraints(model_options, wind, system_variables, parameters, outputs, architecture, scaling)
    cstr_list.append(support_cstr)

    if not model_options['aero']['actuator']['support_only']:
        a_cstr = get_all_model_induction_factor_constraints(model_options, atmos, wind, variables_si, parameters, outputs, architecture, scaling)
        cstr_list.append(a_cstr)

    return cstr_list


def get_all_model_induction_factor_constraints(model_options, atmos, wind, variables, parameters, outputs, architecture, scaling):

    cstr_list = cstr_op.ConstraintList()

    act_comp_labels = actuator_flow.get_actuator_comparison_labels(model_options)

    layer_parent_map = architecture.layer_nodes
    for parent in layer_parent_map:
        for label in act_comp_labels:
            induction_factor_cstr = get_induction_factor_cstr(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label, scaling)
            cstr_list.append(induction_factor_cstr)

        children = architecture.kites_map[parent]
        for kite in children:
            a_assignment_cstr = actuator_flow.get_induction_factor_assignment_cstr(model_options, variables, kite, parent, scaling)
            cstr_list.append(a_assignment_cstr)

    return cstr_list


def get_all_model_support_constraints(model_options, wind, system_variables, parameters, outputs, architecture, scaling):

    variables_si = system_variables['SI']

    cstr_list = cstr_op.ConstraintList()

    for kite in architecture.kite_nodes:
        varrho_and_psi_cstr = actuator_geom.get_varrho_and_psi_cstr(model_options, kite, variables_si, parameters, architecture)
        cstr_list.append(varrho_and_psi_cstr)

    for parent in architecture.layer_nodes:
        bar_varrho_cstr = actuator_geom.get_bar_varrho_cstr(parent, variables_si, architecture, scaling)
        cstr_list.append(bar_varrho_cstr)

        area_cstr = actuator_geom.get_area_cstr(parent, variables_si, parameters, scaling)
        cstr_list.append(area_cstr)

        act_center_cstr = actuator_geom.get_center_cstr(model_options, parent, system_variables, architecture, scaling)
        cstr_list.append(act_center_cstr)

        gamma_cstr = actuator_flow.get_gamma_cstr(parent, variables_si, scaling)
        cstr_list.append(gamma_cstr)

        actuator_orientation_cstr = get_actuator_orientation_cstr(model_options, wind, parent, variables_si, parameters, architecture, scaling)
        cstr_list.append(actuator_orientation_cstr)

        thrust_cstr = actuator_force.get_thrust_cstr(variables_si, outputs, parent, architecture, scaling)
        cstr_list.append(thrust_cstr)

    return cstr_list


def get_induction_factor_cstr(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label, scaling):

    if label == 'qaxi':
        resi = get_momentum_theory_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label, scaling)
    elif label == 'qasym':
        resi = get_steady_asym_pitt_peters_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture, label, scaling)
    elif label == 'uaxi':
        resi = get_unsteady_axi_pitt_peters_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label, scaling)
    elif label == 'uasym':
        resi = get_unsteady_asym_pitt_peters_residual(model_options, atmos, wind, variables, parameters, outputs, parent,
                                               architecture, label, scaling)
    else:
        resi = []
        message = 'model not yet implemented.'
        print_op.log_and_raise_error(message)

    resi_per_parent_length = resi.shape[0]
    a_all_per_parent_length = actuator_flow.get_a_all_var(variables, parent, label).shape[0]
    if (resi_per_parent_length != a_all_per_parent_length):
        message = 'something went wrong with the actuator induction factor constraint, per parent-node, we have ' + str(resi_per_parent_length) + ' residual-lines vs ' + str(a_all_per_parent_length) + ' a-all variables.'
        print_op.log_and_raise_error(message)

    name = 'actuator_induction_factor_' + label + '_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi,
                              name=name,
                              cstr_type='eq')

    return cstr

def get_momentum_theory_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label, scaling):

    a_var = actuator_flow.get_a0_var(variables, parent, label)

    thrust = actuator_force.get_actuator_thrust_var(variables, parent)

    area = actuator_geom.get_area_var(variables, parent)
    qzero = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)
    thrust_den = qzero * area
    # thrust_ref = actuator_coeff.get_thrust_ref(model_options, atmos, wind, parameters)
    # c_t = thrust / thrust_den

    corr_val = actuator_flow.get_corr_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    rhs = 4. * a_var * corr_val

    resi_si = thrust - rhs * thrust_den

    var_type = 'z'
    prefix = actuator_system.get_actuator_var_name_prefix()
    var_name = prefix + 'thrust' + str(parent)
    resi = struct_op.var_si_to_scaled(var_type, var_name, resi_si, scaling)

    return resi


def get_unsteady_axi_pitt_peters_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label, scaling):

    # see page 147 (3.207 - 2.208) of Burton et al, Wind Energy Handbook

    a_all = actuator_flow.get_a_all_var(variables, parent, label)
    da_dt = actuator_flow.get_da_all_var(variables, parent, label)
    a_all_ref = actuator_flow.get_a_all_ref(model_options, parent, scaling)

    c_all, c_numer, c_denom, c_all_ref, c_num_ref, c_den_ref = actuator_coeff.get_c_all_components(model_options, atmos, wind, variables, parameters,
                                                              outputs, parent, architecture, scaling)

    # LL = actuator_coeff.get_LL_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    clockwise_rotation_about_xhat = model_options['aero']['vortex']['clockwise_rotation_about_xhat']
    MM = actuator_coeff.get_MM_matrix(clockwise_rotation_about_xhat)

    # MM is a diagonal matrix; if no wake-misalignment (chi=0) then LLinv is also diagonal.
    term_1 = MM[0, 0] * da_dt[0]
    term_2 = LLinv[0, 0] * a_all[0]
    term_3 = -1. * c_all[0]
    # term_1 = MM[0, 0] * da_dt[0] * c_denom[0] * LLinv_denom
    # term_2 = LLinv_num[0, 0] * a_all[0] * c_denom[0]
    # term_3 = -1. * c_numer[0] * LLinv_denom
    resi_unscaled = term_1 + term_2 + term_3

    # LL_ref = actuator_coeff.get_LL_matrix_ref(model_options,parent, scaling)
    LLinv_ref, LLinv_num_ref, LLinv_denom_ref = actuator_coeff.get_LL_inv_matrix_ref(model_options,parent, scaling)
    term_2_ref = LLinv_ref[0, 0] * a_all_ref[0]
    term_3_ref = c_all_ref[0]
    # term_3_ref = c_num_ref[0] * LLinv_denom_ref
    resi_reference = term_3_ref
    resi = resi_unscaled / vect_op.smooth_abs(resi_reference)
    return resi

def test_signs_of_uaxi(thresh=1e-4):
    # Validation of engineering dynamic inflow models by experimental and numerical approaches
    # W. Yu et al 2016
    # J. Phys.: Conf. Ser. 753 022024
    # https://iopscience.iop.org/article/10.1088/1742-6596/753/2/022024/pdf

    chi = 0.
    time_array = np.linspace(0., 100., 10000)

    rise_time = 0.1

    CT_start = 0.6
    a_start = (1. - np.sqrt(1. - CT_start)) / 2.
    time_switch = 20.

    CT_switch = 0.8
    a_switch = (1. - np.sqrt(1. - CT_switch)) / 2.
    time_return_to_start = 70.

    last_a_before_switch = -99999
    last_a_at_switch_rise_time = -99999
    last_a_before_return_to_start = -999999
    last_a_at_return_to_start_rise_time = -999999

    CT_array = [CT_start]
    a_array = [a_start]
    for tdx in range(1, len(time_array)):
        time = time_array[tdx]
        a_current = a_array[-1]
        if time >= 0 and time < time_switch:
            CT_current = CT_start
        elif time >= time_switch and time < time_return_to_start:
            CT_current = 0.8
        else:
            CT_current = CT_start

        corr = 1. - a_current
        LLinv, _, _ = actuator_coeff.get_LL_inv_matrix_from_corr_and_chi(corr, chi, clockwise_rotation_about_xhat=True)
        MM = actuator_coeff.get_MM_matrix(clockwise_rotation_about_xhat=True)
        da_dt_current = (-LLinv[0, 0] * a_current + CT_current) / MM[0, 0]

        a_found = a_current + da_dt_current * (time - time_array[tdx - 1])

        if time < time_switch:
            last_a_before_switch = a_found
        if time < time_switch + rise_time:
            last_a_at_switch_rise_time = a_found
        if time < time_return_to_start:
            last_a_before_return_to_start = a_found
        if time < time_return_to_start + rise_time:
            last_a_at_return_to_start_rise_time = a_found

        a_array += [a_found]
        CT_array += [CT_current]

    dict = {}
    dict['steady state at start'] = {'found': last_a_before_switch, 'expected': a_start}
    dict['steady state after switch'] = {'found': last_a_before_return_to_start, 'expected': a_switch}

    for name, local_dict in dict.items():
        test = ((local_dict['expected'] - local_dict['found']) / local_dict['expected'])**2. < thresh**2.
        if not test:
            message = 'uaxi gives wrong ' + name
            print_op.log_and_raise_error(message)

    test_2 = (last_a_at_switch_rise_time > a_start) and (last_a_at_switch_rise_time < a_switch)
    if not test_2:
        message = 'uaxi responds in the wrong direction to the switch'
        print_op.log_and_raise_error(message)

    test_3 = (last_a_at_return_to_start_rise_time > a_start) and (last_a_at_return_to_start_rise_time < a_switch)
    if not test_3:
        message = 'uaxi responds in the wrong direction to returning to the start'
        print_op.log_and_raise_error(message)

    # def deltaV(a_local):
    #     return (a_local - a_start) / (a_switch - a_start)
    # deltaV_array = [deltaV(a_local) for a_local in a_array]
    # time_array = np.array(cas.DM(time_array))
    # CT_array = np.array(cas.DM(CT_array))
    # a_array = np.array(cas.DM(a_array))
    # deltaV_array = np.array(cas.DM(deltaV_array))
    #
    # import os
    # import matplotlib
    # if "MPLBACKEND" not in os.environ:
    #     matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plt.plot(time_array, CT_array, label='CT')
    # plt.plot(time_array, a_array, label='a')
    # # notice that deltaV in figure 6 seems to have the sign switched.
    # plt.plot(time_array, deltaV_array, label='deltaV')
    # plt.legend()
    # plt.show()

    return None

def get_unsteady_asym_pitt_peters_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture, label, scaling):

    # see page 147 (3.207 - 2.208) of Burton et al, Wind Energy Handbook

    a_all = actuator_flow.get_a_all_var(variables, parent, label)
    da_dt_all = actuator_flow.get_da_all_var(variables, parent, label)
    a_all_ref = actuator_flow.get_a_all_ref(model_options, parent, scaling)
    da_dt_all_ref = actuator_flow.get_da_all_ref(model_options, parent, scaling)

    c_all, c_numer, c_denom, c_all_ref, c_num_ref, c_den_ref = actuator_coeff.get_c_all_components(model_options, atmos, wind, variables, parameters,
                                                              outputs, parent, architecture, scaling)
    c_denom_diag = cas.diag(c_denom)
    c_denom_diag_ref = cas.diag(c_den_ref)

    # LL = actuator_coeff.get_LL_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    clockwise_rotation_about_xhat = model_options['aero']['vortex']['clockwise_rotation_about_xhat']
    MM = actuator_coeff.get_MM_matrix(clockwise_rotation_about_xhat)
    LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)

    # MM da_dt + LLinv a = c_all
    term_1 = cas.mtimes(MM, da_dt_all)
    term_2 = cas.mtimes(LLinv, a_all)
    term_3 = -1. * c_all
    # term_1 = cas.mtimes(c_denom_diag, cas.mtimes(MM, da_dt_all)) * LLinv_denom
    # term_2 = cas.mtimes(c_denom_diag, cas.mtimes(LLinv_num, a_all))
    # term_3 = -1. * c_numer * LLinv_denom
    resi_unscaled = term_1 + term_2 + term_3

    # LL_ref = actuator_coeff.get_LL_matrix_ref(model_options,parent, scaling)
    LLinv_ref, LLinv_num_ref, LLinv_denom_ref = actuator_coeff.get_LL_inv_matrix_ref(model_options,parent, scaling)
    term_1_ref = cas.mtimes(MM, da_dt_all_ref)
    term_2_ref = cas.mtimes(LLinv_ref, a_all_ref)
    term_3_ref = c_all_ref
    # term_0_ref = actuator_force.get_actuator_thrust_ref(parent, scaling) * cas.DM.ones((3, 1))
    # term_1_ref = cas.mtimes(c_denom_diag_ref, cas.mtimes(MM, da_dt_all_ref)) * LLinv_denom_ref
    # term_2_ref = cas.mtimes(c_denom_diag_ref, cas.mtimes(LLinv_num_ref, a_all_ref))
    # term_3_ref = -1. * c_num_ref * LLinv_denom_ref
    vector_reference = term_2_ref

    resi = []
    for dim in range(3):
        local_scaled = resi_unscaled[dim] / vect_op.smooth_abs(vector_reference[dim])
        resi = cas.vertcat(resi, local_scaled)

    return resi

# def test_signs_of_qasym_with_nonzero_moment(eps=1e-4):
#     # this gives (?) a reasonable result considering the limits of a single-fourier-term-expansion induced velocity
#     # distribution, but it doesn't say anything more than the zero-moment test does. leaving it here, in case something
#     # useful could ever be done with it.
#
#     # Ma L, Li Y, Zhou L, Yang D, Shen X and Du Z (2025)
#     # Study on the aeroelastic performance of 15 MW wind turbine under yaw condition. Front. Energy Res. 13:1571567. doi: 10.3389/fenrg.2025.1571567
#     # https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2025.1571567/full
#     #
#     # "It can be observed that, under yaw condition, the
#     # normal induced velocity on the right half-plane of the rotor for
#     # both rigid and flexible blades is generally higher, particularly in the
#     # spanwise region from 80% to 100% of the blade span. Moreover,
#     # the normal induced velocity at 90° azimuth is significantly greater
#     # than at 270°,"
#
#     u_zero = 10.59 # m/s, page 03
#     diameter = 240. #m, page 03
#     gamma_yaw = 20. * np.pi / 180. #pg 05
#     hub_radius = 0.2 * (diameter/2.) # fig. 8, pg. 8
#
#     area = np.pi * (diameter / 2.)**2.
#     rho = 1.225 #guessed
#     actuator_dynamic_pressure = 0.5 * rho * u_zero**2.
#     exterior_radius = diameter/2.
#     radius_bar = (hub_radius + exterior_radius) / 2.
#     wingspan = (exterior_radius - radius_bar) * 2.
#     clockwise_rotation_about_xhat = True
#
#     # # values roughly estimated (by eye) from Figure 15
#     # thrust = 2e6 #N
#     # tilt_moment = 6e6 #Nm
#     # yaw_moment = -6.5e6 #Nm
#
#     # values roughly estimated (by eye) from Figure 5
#     thrust = 2.28e6 #N
#     tilt_moment =2.2e6 #Nm
#     yaw_moment = -3e6 #Nm
#
#     tilt_angle = 6. * np.pi / 180. # https://docs.nlr.gov/docs/fy20osti/75698.pdf
#
#     n_vec = vect_op.xhat_dm() + cas.tan(gamma_yaw) * vect_op.yhat_dm() + cas.tan(tilt_angle) * vect_op.zhat_dm()
#     n_hat = vect_op.normalize(n_vec)
#     u_hat = vect_op.xhat_dm()
#     dcm_act, dcm_wind = actuator_flow.construct_actuator_and_wind_reference_frames_from_n_and_u(n_hat, u_hat)
#     y_rotor_hat = dcm_act[:, 1]
#     z_rotor_hat = dcm_act[:, 2]
#
#     moment_y = tilt_moment
#     moment_z = yaw_moment
#
#     approx_avg_normal_ind_velocity = (-2.5 + (-6.)) / 2.
#     approx_a0 = -1. * approx_avg_normal_ind_velocity / u_zero
#
#     c_all, c_numer, c_denom = actuator_coeff.assemble_c_components_from_pieces(thrust, moment_y, moment_z, actuator_dynamic_pressure, area, radius_bar, wingspan)
#
#     chi = actuator_flow.get_wake_angle_chi_jimenez(gamma_yaw, thrust, area, actuator_dynamic_pressure)
#     corr = actuator_flow.get_corr_val_coleman(approx_a0, gamma_yaw, chi)
#     LL_matr = actuator_coeff.get_LL_matrix_from_corr_and_chi(corr, chi, clockwise_rotation_about_xhat=True)
#
#     a_all = cas.mtimes(LL_matr, c_all)
#     a_dict = actuator_flow.unstack_a_all(a_all)
#
#     dict = {}
#     idx = 0
#     for mu in np.linspace(hub_radius/exterior_radius, 1., 10):
#         for psi in np.linspace(0., 2. * np.pi, 13):
#
#             radius = mu * exterior_radius
#             radial_vector = actuator_geom.assemble_radial_vector(radius, y_rotor_hat, z_rotor_hat, psi, clockwise_rotation_about_xhat)
#             a_local = actuator_flow.assemble_asym_local_induction_factor(a_dict, psi, mu)
#             ui = u_zero * (-1.) * a_local
#
#             local_dict = {}
#             local_dict['mu'] = float(mu)
#             local_dict['psi'] = float(psi)
#             local_dict['y'] = float(cas.mtimes(radial_vector.T, y_rotor_hat))
#             local_dict['z'] = float(cas.mtimes(radial_vector.T, z_rotor_hat))
#             local_dict['a'] = float(a_local)
#             local_dict['ui'] = float(ui)
#             dict[idx] = local_dict
#             idx += 1
#
#     y_array = np.array([local_dict['y'] for local_dict in dict.values()])
#     z_array = np.array([local_dict['z'] for local_dict in dict.values()])
#     # a_array = np.array([local_dict['a'] for local_dict in dict.values()])
#     ui_array = np.array([local_dict['ui'] for local_dict in dict.values()])
#
#     import os
#     import matplotlib
#     if "MPLBACKEND" not in os.environ:
#         matplotlib.use("TkAgg")
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots()
#     contour = ax.tricontourf(y_array, z_array, ui_array, levels=30, cmap='RdYlBu_r')
#     fig.colorbar(contour, ax=ax, label='Normal induced velocity [m/s]')
#     ax.set_xlabel('y')
#     ax.set_ylabel('z')
#     ax.set_aspect('equal')
#     plt.show()
#
#     return None

def test_signs_of_qasym_with_zero_moment(eps=1e-3):
    # see Figures in Burton et al, Wind Energy Handbook:
    # axes in figure 3.46, page 97
    # streamline sketch in figure 3.54, page 105,
    # angle-of-attack sketch in figure 3.61 page 112
    # if positive gamma rotates about +zhat, then high force on side with rotor into wind and low force on side with rotor out of wind,
    # meaning torque rotates turbine about -zhat

    # "If the yaw error is positive, the left-hand side of the rotor (looking downwind) which is the downwind side of the rotor, should experience lower induction factors than the right-hand side of the rotor (upwind side)." - Suzuki2000, page 22

    # values in test roughly inspired by values in fig. 4, page 8, of
    # A review of momentum models for the actuator disk in yaw, Hur et al,
    # https://arc.aiaa.org/doi/10.2514/6.2019-1799
    # AIAA 2019-1799
    # Session: Wind Energy Modeling Approaches
    # Published Online:6 Jan 2019 https://doi.org/10.2514/6.2019-1799
    # https://www.researchgate.net/profile/Chihoon-Hur-2/publication/330199212_A_review_of_momentum_models_for_the_actuator_disk_in_yaw/links/5d60078c299bf1f70b077ec2/A-review-of-momentum-models-for-the-actuator-disk-in-yaw.pdf

    ct = 0.6
    approx_a0 = 0.18
    gamma_yaw = 30. * np.pi / 180.

    cmz = 0.
    cmy = 0.

    exterior_radius = 1.
    radius_bar = 0.7 * exterior_radius # Page 199 of branlard
    wingspan = (exterior_radius - radius_bar) * 2.

    area = np.pi * exterior_radius**2.
    rho = 1.225
    u_zero = 5.
    actuator_dynamic_pressure = 0.5 * rho * u_zero**2.
    clockwise_rotation_about_xhat = True

    thrust = ct * actuator_dynamic_pressure * area
    moment_y = cmy * actuator_dynamic_pressure * area * exterior_radius
    moment_z = cmz * actuator_dynamic_pressure * area * exterior_radius

    c_all, c_numer, c_denom = actuator_coeff.assemble_c_components_from_pieces(thrust, moment_y, moment_z, actuator_dynamic_pressure, area, radius_bar, wingspan)
    c_denom_diag_inv = cas.inv(cas.diag(c_denom))

    corr = (1. - approx_a0)
    chi = (0.6 * approx_a0 + 1.) * gamma_yaw
    LL_matr = actuator_coeff.get_LL_matrix_from_corr_and_chi(corr, chi, clockwise_rotation_about_xhat=True)

    # a_all = cas.mtimes(c_denom_diag_inv, cas.mtimes(LL_matr, c_numer))
    a_all = cas.mtimes(LL_matr, c_all)
    a_dict = actuator_flow.unstack_a_all(a_all)

    n_hat = cas.cos(gamma_yaw) * vect_op.xhat_dm() + cas.sin(gamma_yaw) * vect_op.yhat_dm()
    u_hat = vect_op.xhat_dm()
    dcm_act, dcm_wind = actuator_flow.construct_actuator_and_wind_reference_frames_from_n_and_u(n_hat, u_hat)
    y_rotor_hat = dcm_act[:, 1]
    z_rotor_hat = dcm_act[:, 2]

    dict = {}
    idx = 0
    smallest = +999999.
    largest = -999999.
    for mu in [1.]:
        for psi in [0., np.pi/2., np.pi, 3. * np.pi/2., 2. * np.pi]:

            radius = mu * exterior_radius
            radial_vector = actuator_geom.assemble_radial_vector(radius, y_rotor_hat, z_rotor_hat, psi, clockwise_rotation_about_xhat)
            a_local = actuator_flow.assemble_asym_local_induction_factor(a_dict, psi, mu)

            if cas.mtimes(radial_vector.T, y_rotor_hat)**2. < eps:
                tentative_intermediate_value = a_local
                stream_origin = cas.mtimes(radial_vector.T, u_hat)

            if a_local < smallest:
                smallest = a_local
            if a_local > largest:
                largest = a_local

            local_point = {'y': cas.mtimes(radial_vector.T, y_rotor_hat), 'z': cas.mtimes(radial_vector.T, z_rotor_hat), 'u': cas.mtimes(radial_vector.T, u_hat), 'a': a_local}
            dict[idx] = local_point
            idx += 1

    # test -1 = positive y values are upstream of negative y values
    pos_y_upstream_of_neg_y = True

    # test_0 = tentative intermediate value is actually intermediate
    intermediate_value_is_intermediate = (smallest < tentative_intermediate_value) and (largest > tentative_intermediate_value)

    # test_1 = when y is zero we actually have intermediate value
    when_y_is_zero_we_have_intermediate_value = True
    for idx, local_dict in dict.items():
        if local_dict['y']**2. < eps:
            when_y_is_zero_we_have_intermediate_value = when_y_is_zero_we_have_intermediate_value and ((local_dict['a'] - tentative_intermediate_value)**2. < eps)

    # test 2 = when y is positive, we have the larger induction factor
    when_y_is_pos_we_have_largest_val = True
    for idx, local_dict in dict.items():
        if local_dict['y'] > eps:
            when_y_is_pos_we_have_largest_val = when_y_is_pos_we_have_largest_val and ((local_dict['a'] - largest)**2. < eps)
            pos_y_upstream_of_neg_y = pos_y_upstream_of_neg_y and (local_dict['u'] < stream_origin)

    # test 3 = when y is negative, we have the smaller induction factor
    when_y_is_neg_we_have_smallest_val = True
    for idx, local_dict in dict.items():
        if local_dict['y'] < (-1 * eps):
            when_y_is_neg_we_have_smallest_val = when_y_is_neg_we_have_smallest_val and ((local_dict['a'] - smallest)**2. < eps)
            pos_y_upstream_of_neg_y = pos_y_upstream_of_neg_y and (local_dict['u'] > stream_origin)

    # test 4 = values are roughly similar to those printed in Hur et al
    expected = cas.DM([0.09, 0.18, 0.26])
    found = cas.DM([smallest, tentative_intermediate_value, largest])
    diff = expected - found
    values_are_roughly_similar = (cas.mtimes(diff.T, diff)) < (eps * vect_op.norm(expected))

    test = pos_y_upstream_of_neg_y and intermediate_value_is_intermediate and when_y_is_zero_we_have_intermediate_value and when_y_is_pos_we_have_largest_val and when_y_is_neg_we_have_smallest_val and values_are_roughly_similar

    if not test:
        print_op.print_dict_as_table(dict)
        message = 'something went wrong with the sign conventions on the yawed/skewed-wake model.'
        print_op.log_and_raise_error(message)
    return None


def get_steady_asym_pitt_peters_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture, label, scaling):

    c_all, c_numer, c_denom, c_all_ref, c_num_ref, c_den_ref = actuator_coeff.get_c_all_components(model_options, atmos, wind, variables, parameters, outputs, parent, architecture, scaling)
    c_denom_diag = cas.diag(c_denom)
    c_denom_diag_ref = cas.diag(c_den_ref)

    LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_val(model_options, atmos, wind, variables, outputs,
                                                                         parameters, parent, architecture, label)
    LL_matr = actuator_coeff.get_LL_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    LL_ref = actuator_coeff.get_LL_matrix_ref(model_options, parent, scaling)

    a_all = actuator_flow.get_a_all_var(variables, parent, label)
    a_all_ref = actuator_flow.get_a_all_ref(model_options, parent, scaling)

    # see page 136 (3.188) of Burton et al, Wind Energy Handbook
    term_1 = -1. * a_all
    term_2 = cas.mtimes(LL_matr, c_all)
    # term_1 = -1. * cas.mtimes(c_denom_diag, a_all)
    # term_2 = cas.mtimes(LL_matr, c_numer)
    resi_unscaled = term_1 + term_2

    term_0_ref = actuator_force.get_actuator_thrust_ref(parent, scaling) * cas.DM.ones((3, 1))
    term_1_ref = a_all_ref
    term_2_ref = cas.mtimes(LL_ref, c_all_ref)
    # term_1_ref = -1. * cas.mtimes(c_denom_diag_ref, a_all_ref)
    # term_2_ref = cas.mtimes(LL_ref, c_num_ref)

    # vector_reference = term_2_ref
    # print('qasym term_2_ref')
    vector_reference = term_1_ref
    # print('qasym term_1_ref')

    resi = []
    for dim in range(3):
        local_scaled = resi_unscaled[dim] / vect_op.smooth_abs(vector_reference[dim])
        resi = cas.vertcat(resi, local_scaled)

    return resi

def get_actuator_orientation_cstr(model_options, wind, parent, variables_si, parameters, architecture, scaling):

    # --------------------
    # 2 * dcms (18 vars) + 3 * lengths (3 vars)
    # = 21 variables total
    # --------------------
    # 21 constraints total

    cstr_list = cstr_op.ConstraintList()

    # 9
    uhat_cstr = actuator_flow.get_wind_dcm_u_along_uzero_cstr(model_options, wind, parent, variables_si, architecture, scaling)
    cstr_list.append(uhat_cstr)
    wind_dcm_ortho_cstr = actuator_flow.get_wind_dcm_ortho_cstr(parent, variables_si)
    cstr_list.append(wind_dcm_ortho_cstr)

    # 9
    nhat_cstr = actuator_geom.get_act_dcm_n_along_normal_cstr(model_options, parent, variables_si, architecture, scaling)
    cstr_list.append(nhat_cstr)
    act_dcm_ortho_cstr = actuator_geom.get_act_dcm_ortho_cstr(parent, variables_si)
    cstr_list.append(act_dcm_ortho_cstr)

    # 3
    what_cstr = actuator_flow.get_wind_dcm_w_along_act_dcm_z_cstr(model_options, wind, parent, variables_si, architecture, scaling)
    cstr_list.append(what_cstr)

    number_constraints = cstr_list.get_expression_list('eq').shape[0]
    if number_constraints != 21:
        message = 'there are an inappropriate number of actuator orientation constraints (' + str(number_constraints) + ')'
        print_op.log_and_raise_error(message)

    return cstr_list



def collect_actuator_outputs(model_options, atmos, wind, variables, outputs, parameters, architecture, scaling):
    outputs = collect_actuator_support_outputs(model_options, atmos, wind, variables, outputs, parameters, architecture, scaling)
    outputs = collect_actuator_induction_factor_outputs(model_options, variables, outputs, architecture)
    return outputs


def collect_actuator_induction_factor_outputs(model_options, variables, outputs, architecture):

    kite_nodes = architecture.kite_nodes
    act_comp_labels = actuator_flow.get_actuator_comparison_labels(model_options)

    if 'actuator' not in list(outputs.keys()):
        outputs['actuator'] = {}

    for kite in kite_nodes:
        parent = architecture.parent_map[kite]
        for label in act_comp_labels:

            if model_options['aero']['actuator']['support_only']:
                local_a = cas.DM(0.)
            else:
                local_a = actuator_flow.get_local_induction_factor(model_options, variables, kite, parent, label)
            outputs['actuator']['local_a_' + label + str(kite)] = local_a

        current_label = actuator_flow.get_label(model_options)
        outputs['actuator']['local_a' + str(kite)] = outputs['actuator']['local_a_' + current_label + str(kite)]

    layer_parents = architecture.layer_nodes
    for parent in layer_parents:
        for label in act_comp_labels:

            if model_options['aero']['actuator']['support_only']:
                a0_var = cas.DM(0.)
            else:
                a0_var = actuator_flow.get_a0_var(variables, parent, label)
            outputs['actuator']['a0_' + label + str(parent)] = a0_var

    return outputs


def collect_actuator_support_outputs(model_options, atmos, wind, variables, outputs, parameters, architecture, scaling):

    kite_nodes = architecture.kite_nodes
    clockwise_rotation_about_xhat = model_options['aero']['vortex']['clockwise_rotation_about_xhat']

    if 'actuator' not in list(outputs.keys()):
        outputs['actuator'] = {}

    for kite in kite_nodes:
        outputs['actuator']['rhat' + str(kite)] = actuator_geom.get_kite_radial_vector(kite, variables, architecture, clockwise_rotation_about_xhat)
        outputs['actuator']['radius' + str(kite)] = actuator_geom.get_kite_radius(kite, variables, architecture, parameters)

    layer_parents = architecture.layer_nodes
    for parent in layer_parents:

        center = actuator_geom.get_actuator_position_var(variables, parent)
        velocity = geom.get_center_velocity(model_options, parent, variables, architecture)
        area = actuator_geom.get_area_var(variables, parent)
        avg_radius = actuator_geom.get_average_radius(variables, parent, architecture, parameters)
        nhat = actuator_system.get_actuator_vector_unit_var(variables, 'n', parent)
        yaw_angle = actuator_flow.get_gamma_var(variables, parent)
        q_app = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)

        outputs['actuator']['center' + str(parent)] = center
        outputs['actuator']['velocity' + str(parent)] = velocity
        outputs['actuator']['area' + str(parent)] = area
        outputs['actuator']['avg_radius' + str(parent)] = avg_radius
        outputs['actuator']['nhat' + str(parent)] = nhat
        outputs['actuator']['bar_varrho' + str(parent)] = actuator_geom.get_bar_varrho_var(variables, parent)
        outputs['actuator']['yaw' + str(parent)] = yaw_angle
        outputs['actuator']['yaw_deg' + str(parent)] = yaw_angle * 180. / np.pi
        outputs['actuator']['dyn_pressure' + str(parent)] = q_app

        thrust = actuator_force.get_actuator_thrust_var(variables, parent)
        ct = actuator_coeff.get_ct_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture)
        outputs['actuator']['thrust' + str(parent)] = thrust
        outputs['actuator']['ct' + str(parent)] = ct

        outputs['actuator']['w_z_check' + str(parent)] = actuator_flow.get_wzero_parallel_z_rotor_check(variables, parent)
        outputs['actuator']['gamma_check' + str(parent)] = actuator_flow.get_gamma_check(variables, parent)
        outputs['actuator']['gamma_comp' + str(parent)] = actuator_flow.get_gamma_val(variables, parent)

    return outputs


def draw_actuator(ax, side, plot_dict, cosmetics, index):
    if 'actuator' in plot_dict['interpolation_si']['outputs'].keys():
        actuator_geom.draw_actuator_geometry(ax, side, plot_dict, cosmetics, index)
        actuator_flow.draw_actuator_flow(ax, side, plot_dict, cosmetics, index)
    return None

def independent_test():
    actuator_flow.test_gamma_angle_for_dcms()
    actuator_coeff.test_LL_inversion_correctly_implemented()
    test_signs_of_uaxi()
    test_signs_of_qasym_with_zero_moment()
    # test_signs_of_qasym_with_nonzero_moment()
    return None


def sanity_check(init_options, variables_si, wind, architecture, epsilon=1e-5):
    independent_test()

    for parent in architecture.layer_nodes:
        actuator_flow.check_that_wzero_is_parallel_to_z_rotor(variables_si, parent, epsilon)
        actuator_flow.check_that_uzero_has_positive_component_in_dominant_wind_direction(wind, variables_si, parent, epsilon)
        if init_options['induction']['normal_vector_model'] != 'xhat':
            actuator_flow.check_that_gamma_is_consistent(variables_si, parent, epsilon)
        actuator_geom.check_that_actuator_center_is_above_minimum_altitude(init_options, variables_si, parent)
        actuator_geom.check_that_varrho_initialization_is_set_and_averages_correctly(architecture, variables_si, parent)

    return None

if __name__ == "__main__":
    independent_test()