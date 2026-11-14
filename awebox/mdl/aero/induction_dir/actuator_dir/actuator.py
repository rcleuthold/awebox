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
        a_cstr = get_all_model_induction_factor_constraints(model_options, atmos, wind, system_variables, parameters, outputs, architecture, scaling)
        cstr_list.append(a_cstr)

    return cstr_list


def get_all_model_induction_factor_constraints(model_options, atmos, wind, system_variables, parameters, outputs, architecture, scaling):

    variables_si = system_variables['SI']

    cstr_list = cstr_op.ConstraintList()

    act_comp_labels = actuator_flow.get_actuator_comparison_labels(model_options)

    layer_parent_map = architecture.layer_nodes
    for parent in layer_parent_map:
        for label in act_comp_labels:
            induction_factor_cstr = get_induction_factor_cstr(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, label, scaling)
            cstr_list.append(induction_factor_cstr)

        children = architecture.kites_map[parent]
        for kite in children:
            a_assignment_cstr = actuator_flow.get_induction_factor_assignment_cstr(model_options, variables_si, kite, parent, scaling)
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

        actuator_orientation_cstr = get_actuator_orientation_cstr(model_options, wind, parent, system_variables, parameters, architecture, scaling)
        cstr_list.append(actuator_orientation_cstr)

        thrust_cstr = actuator_force.get_thrust_cstr(variables_si, outputs, parent, architecture, scaling)
        cstr_list.append(thrust_cstr)

    return cstr_list


def get_induction_factor_cstr(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, label, scaling):
    variables_si = system_variables['SI']

    if label == 'qaxi':
        resi = get_momentum_theory_residual(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, label, scaling)
    elif label == 'qasym':
        resi = get_steady_asym_pitt_peters_residual(model_options, atmos, wind, system_variables, parameters, outputs, parent, architecture, label, scaling)
    elif label == 'uaxi':
        resi = get_unsteady_axi_pitt_peters_residual(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, label, scaling)
    elif label == 'uasym':
        resi = get_unsteady_asym_pitt_peters_residual(model_options, atmos, wind, system_variables, parameters, outputs, parent, architecture, label, scaling)
    else:
        resi = []
        message = 'model not yet implemented.'
        print_op.log_and_raise_error(message)

    resi_per_parent_length = resi.shape[0]
    a_all_per_parent_length = actuator_flow.get_a_all_var(variables_si, parent, label).shape[0]
    if (resi_per_parent_length != a_all_per_parent_length):
        message = 'something went wrong with the actuator induction factor constraint, per parent-node, we have ' + str(resi_per_parent_length) + ' residual-lines vs ' + str(a_all_per_parent_length) + ' a-all variables.'
        print_op.log_and_raise_error(message)

    name = 'actuator_induction_factor_' + label + '_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi,
                              name=name,
                              cstr_type='eq')

    return cstr

def scale_resi(resi_unscaled, resi_reference, term_ref_name):

    message = 'actuator model being scaled according to ' + term_ref_name
    print_op.base_print(message, level='info')

    resi = []
    for dim in range(resi_reference.shape[0]):
        local_scaled = resi_unscaled[dim] / vect_op.smooth_abs(resi_reference[dim])
        resi = cas.vertcat(resi, local_scaled)
    return resi

def assemble_pitt_peters_style_residual(a_all, da_dt_all, t_star, t_star_num, t_star_den, MM, LLinv, LLinv_num, LLinv_denom, c_all, c_numer, c_denom):

    # see page 147 (3.207 - 2.208) of Burton et al, Wind Energy Handbook

    # term_1_standard = cas.mtimes(MM, da_dt_all) * t_star
    # term_2_stardard = cas.mtimes(LLinv, a_all)
    # term_3_standard = -1. * c_all

    MM_rearranged = LLinv_denom * cas.mtimes(cas.diag(c_denom), MM)
    LLinv_rearranged = cas.mtimes(cas.diag(c_denom), LLinv_num)
    c_rearranged = LLinv_denom * c_numer

    term_1_rearranged = cas.mtimes(MM_rearranged, da_dt_all) * t_star_num
    term_2_rearranged = cas.mtimes(LLinv_rearranged, a_all) * t_star_den
    term_3_rearranged = -1. * c_rearranged * t_star_den

    resi = term_1_rearranged + term_2_rearranged + term_3_rearranged
    return resi, term_1_rearranged, term_2_rearranged, term_3_rearranged

def test_momentum_theory_in_helicopter_conditions(thresh=0.08):

    all_heli_dict = {}

    # https://kumar-sumeet.github.io/HeliAeroNotes/Lectures/1_MT/MT.html
    # digitized from plot
    all_heli_dict['R22'] = {'disk_loading_lb_ft2': 2.6269,
                            'far_wake_downwash_knots': 27.106}
    all_heli_dict['AH-1G'] = {'disk_loading_lb_ft2': 5.2771,
                              'far_wake_downwash_knots': 39.889}
    all_heli_dict['UH-60'] = {'disk_loading_lb_ft2': 7.1936,
                              'far_wake_downwash_knots': 47.004}
    all_heli_dict['AH-64'] = {'disk_loading_lb_ft2': 8.2153,
                              'far_wake_downwash_knots': 50.122}
    # all_heli_dict['XV-15'] = {'disk_loading_lb_ft2': 13.5082, # not included because dual-rotor
    #                           'far_wake_downwash_knots': 63.002}
    all_heli_dict['CH-53E'] = {'disk_loading_lb_ft2': 14.1743,
                             'far_wake_downwash_knots': 64.495}
    # all_heli_dict['V-22'] = {'disk_loading_lb_ft2': 21.8021, # not included, because dual-rotor
    #                           'far_wake_downwash_knots': 80.623}

    all_heli_dict['R22']['disk_area_m2'] = 46.2     # https://en.wikipedia.org/wiki/Disk_loading
    all_heli_dict['CH-53E']['disk_area_m2'] = 460.     # https://en.wikipedia.org/wiki/Disk_loading
    all_heli_dict['AH-1G']['disk_area_m2'] = 141. #https://www.helis.com/database/model/AH-1G-Cobra/
    all_heli_dict['UH-60']['disk_area_m2'] = 210. #https://en.wikipedia.org/wiki/Sikorsky_UH-60_Black_Hawk
    all_heli_dict['AH-64']['disk_area_m2'] = 14.63 #https://en.wikipedia.org/wiki/Boeing_AH-64_Apache

    u_zero = 1.
    rho = 1.225
    hub_radius = 0.

    lb_ft2_to_pa = 47.88026
    knot_to_mps = 0.514444
    for heli_dict in all_heli_dict.values():
        heli_dict['disk_loading_pa'] = heli_dict['disk_loading_lb_ft2'] * lb_ft2_to_pa
        heli_dict['far_wake_downwash_mps'] = heli_dict['far_wake_downwash_knots'] * knot_to_mps
        heli_dict['exterior_radius'] = (heli_dict['disk_area_m2'] / np.pi)**0.5
        heli_dict['thrust_up_n'] = heli_dict['disk_loading_pa'] * heli_dict['disk_area_m2']

        exterior_radius = heli_dict['exterior_radius']
        area = heli_dict['disk_area_m2']

        actuator_dynamic_pressure = 0.5 * rho * u_zero**2.
        radius_bar = (hub_radius + exterior_radius) / 2.
        wingspan = (exterior_radius - radius_bar) * 2.

        thrust = -1. * heli_dict['thrust_up_n'] # nhat is defined from upstream-to-downstream, eg: downwards
        moment_y = 0.
        moment_z = 0.

        n_hat = -vect_op.zhat_dm()
        u_hat = vect_op.xhat_dm()
        dcm_act, dcm_wind = actuator_flow.construct_actuator_and_wind_reference_frames_from_n_and_u(n_hat, u_hat)
        y_rotor_hat = dcm_act[:, 1]
        z_rotor_hat = dcm_act[:, 2]

        # n_hat = -vect_op.zhat_dm()
        # u_hat = n_hat
        # w_hat = vect_op.yhat_dm()
        # z_rotor_hat = w_hat
        # y_rotor_hat = vect_op.cross(z_rotor_hat, n_hat)
        # v_hat = vect_op.cross(w_hat, u_hat)
        # dcm_act = cas.horzcat(n_hat, y_rotor_hat, z_rotor_hat)
        # dcm_wind = cas.horzcat(u_hat, v_hat, w_hat)

        gamma_yaw = vect_op.angle_between(u_hat, n_hat, z_rotor_hat)

        approx_a0 = 0.5 - (1 - (thrust / actuator_dynamic_pressure / area))**0.5 / 2.

        c_all, c_numer, c_denom = actuator_coeff.assemble_c_components_from_pieces(thrust, moment_y, moment_z, actuator_dynamic_pressure, area, radius_bar, wingspan)

        step_history = {0: {'a0': approx_a0, 'acos': 0, 'asin': 0}}
        for step in range(20):
            corr = actuator_flow.get_corr_val_glauert(approx_a0, gamma_yaw)
            chi = actuator_flow.get_wake_angle_chi_coleman(gamma_yaw, approx_a0)
            LL_matr = actuator_coeff.get_LL_matrix_from_corr_and_chi(corr, chi)

            a_all = cas.mtimes(LL_matr, c_all)
            a_dict = actuator_flow.unstack_a_all(a_all)
            step_history[step] = a_dict

            approx_a0 = a_dict['a0']

        vec_u_ind = -approx_a0 * n_hat
        heli_dict['a0'] = approx_a0
        heli_dict['vec u_ind at disk [m/s]'] = vec_u_ind
        heli_dict['vec u_ind far-wake [m/s]'] = 2. * vec_u_ind
        heli_dict['far-wake u_ind downwards [knots]'] = cas.mtimes(2. * vec_u_ind.T, -vect_op.zhat_dm()) / knot_to_mps

        diff = (heli_dict['far-wake u_ind downwards [knots]'] - heli_dict['far_wake_downwash_knots'])
        error = diff / heli_dict['far_wake_downwash_knots']
        if error**2. > thresh**2.:
            message = 'acutator model q.axi. / momentum theory application to helicopters seems not to have worked correctly. '
            message += 'local error found is ' + str(error)
            print_op.log_and_raise_error(message)

    # print_op.print_dict_as_table(all_heli_dict)

    # base_name = "helicopter_momentum_theory_verification"
    #
    # import os
    # import matplotlib
    # if "MPLBACKEND" not in os.environ:
    #     matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(layout='constrained')
    # for heli_name, heli_dict in all_heli_dict.items():
    #     ax.scatter([float(heli_dict['disk_loading_lb_ft2'])], [float(heli_dict['far-wake u_ind downwards [knots]'])], label=heli_name)
    # ax.legend()
    # ax.set_xlim(0, 22)
    # ax.set_ylim(0, 80)
    # # ax.set_xlabel('lb_ft2')
    # # ax.set_ylabel('knots')
    #
    # # 1. Secondary X-axis: Convert Seconds to Milliseconds (multiply by 1000)
    # # We provide a forward and backward function for the mapping
    # to_pa = lambda x: x * lb_ft2_to_pa
    # from_pa = lambda x: x / lb_ft2_to_pa
    # sec_x = ax.secondary_xaxis('top', functions=(to_pa, from_pa))
    # sec_x.set_xlabel('Disk Loading [Pa]')
    #
    # # 2. Secondary Y-axis: Convert Meters to Feet (multiply by 3.28084)
    # to_mps = lambda y: y * knot_to_mps
    # from_mps = lambda y: y / knot_to_mps
    # sec_y = ax.secondary_yaxis('right', functions=(to_mps, from_mps))
    # sec_y.set_ylabel('Induced Speed (Vertically-downwards) [m/s]')
    #
    # top_ticks_secondary = np.linspace(0, 1000, 6)
    # right_ticks_secondary = np.linspace(0, 40, 9)
    # # 2. Use your inverse functions to map them back to primary coordinates
    # # (Using the lambda logic from your code)
    # from_pa = lambda x: x / lb_ft2_to_pa  # Inverse function for X
    # from_mps = lambda y: y / knot_to_mps  # Inverse function for Y
    # ax.set_xticks([from_pa(x) for x in top_ticks_secondary])
    # ax.set_yticks([from_mps(y) for y in right_ticks_secondary])
    # # 3. Turn on the primary grid (it will now align with your custom spacing)
    # ax.grid(True, linestyle='--', alpha=0.5)
    #
    # plt.savefig(base_name + ".pdf", format="pdf")

    return None


def get_momentum_theory_residual(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, label, scaling):
    variables_si = system_variables["SI"]

    a_all = actuator_flow.get_a_all_var(variables_si, parent, label)
    a_all_ref = actuator_flow.get_a_all_ref(model_options, parent, scaling)

    c_all, c_numer, c_denom, c_all_ref, c_num_ref, c_den_ref = actuator_coeff.get_c_all_components(model_options, atmos, wind, system_variables, parameters, outputs, parent, architecture, scaling)
    LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_val(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, scaling, label)
    t_star, t_star_num, t_star_den = actuator_coeff.get_t_star(variables_si, parameters, parent)

    resi_unscaled, _, _, _ = assemble_pitt_peters_style_residual(a_all, 0., t_star, t_star_num, t_star_den, 0., LLinv[0, 0], LLinv_num[0, 0], LLinv_denom, c_all[0], c_numer[0], c_denom[0])

    LLinv_ref, LLinv_num_ref, LLinv_denom_ref = actuator_coeff.get_LL_inv_matrix_ref(model_options, parent, scaling)
    t_star_ref, t_star_num_ref, t_star_den_ref = actuator_coeff.get_t_star_ref(model_options, parameters, parent, scaling)
    _, _, term_2_ref, term_3_ref = assemble_pitt_peters_style_residual(a_all_ref, 0., t_star_ref, t_star_num_ref, t_star_den_ref, 0., LLinv_ref[0, 0], LLinv_num_ref[0, 0], LLinv_denom_ref, c_all_ref[0], c_num_ref[0], c_den_ref[0])

    resi_reference = term_2_ref
    term_ref_name = 'term_2'

    # resi_reference = term_3_ref
    # term_ref_name = 'term_3'

    resi = scale_resi(resi_unscaled, resi_reference, term_ref_name)
    return resi


def get_unsteady_axi_pitt_peters_residual(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, label, scaling):
    variables_si = system_variables['SI']

    a_all = actuator_flow.get_a_all_var(variables_si, parent, label)
    da_dt_all = actuator_flow.get_da_all_var(variables_si, parent, label)
    a_all_ref = actuator_flow.get_a_all_ref(model_options, parent, scaling)
    da_dt_all_ref = actuator_flow.get_da_all_ref(model_options, parent, scaling)

    c_all, c_numer, c_denom, c_all_ref, c_num_ref, c_den_ref = actuator_coeff.get_c_all_components(model_options, atmos, wind, system_variables, parameters, outputs, parent, architecture, scaling)

    LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_val(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, scaling, label)
    MM = actuator_coeff.get_MM_matrix()
    t_star, t_star_num, t_star_den = actuator_coeff.get_t_star(variables_si, parameters, parent)

    resi_unscaled, _, _, _ = assemble_pitt_peters_style_residual(a_all, da_dt_all, t_star, t_star_num, t_star_den, MM[0, 0], LLinv[0, 0], LLinv_num[0, 0], LLinv_denom, c_all[0], c_numer[0], c_denom[0])

    LLinv_ref, LLinv_num_ref, LLinv_denom_ref = actuator_coeff.get_LL_inv_matrix_ref(model_options, parent, scaling)
    t_star_ref, t_star_num_ref, t_star_den_ref = actuator_coeff.get_t_star_ref(model_options, parameters, parent, scaling)

    _, term_1_ref, term_2_ref, term_3_ref = assemble_pitt_peters_style_residual(a_all_ref, da_dt_all_ref, t_star_ref, t_star_num_ref, t_star_den_ref, MM[0, 0], LLinv_ref[0, 0], LLinv_num_ref[0, 0], LLinv_denom_ref, c_all_ref[0], c_num_ref[0], c_den_ref[0])

    # resi_reference = term_1_ref
    # term_ref_name = 'term_1'

    resi_reference = term_2_ref
    term_ref_name = 'term_2'

    # resi_reference = term_3_ref
    # term_ref_name = 'term_3'

    resi = scale_resi(resi_unscaled, resi_reference, term_ref_name)
    return resi

def test_signs_of_uaxi(thresh=1e-4):
    # Validation of engineering dynamic inflow models by experimental and numerical approaches
    # W. Yu et al 2016
    # J. Phys.: Conf. Ser. 753 022024
    # https://iopscience.iop.org/article/10.1088/1742-6596/753/2/022024/pdf

    chi = 0.

    diameter = 600e-3 #m, page 4
    air_density = 1.225 #kg/m^3
    u_zero_mag = 34 #m/s

    actuator_dynamic_pressure = 0.5 * air_density * u_zero_mag**2.
    exterior_radius = diameter / 2.
    actuator_area = np.pi * exterior_radius**2.
    t_star, t_star_num, t_star_den = actuator_coeff.compute_t_star_from_exterior_radius_and_actuator_apparent_speed(exterior_radius, u_zero_mag)

    observation_tau = 0.1

    tau_start = -20.
    tau_array = np.linspace(tau_start, 100. , 10000)

    CT_start = 0.6
    a_start = (1. - np.sqrt(1. - CT_start)) / 2.
    tau_switch = 20 + tau_start

    CT_switch = 0.8
    a_switch = (1. - np.sqrt(1. - CT_switch)) / 2.
    tau_return_to_start = 70. + tau_start

    last_a_before_switch = -99999
    last_a_at_switch_rise_time = -99999
    last_a_before_return_to_start = -999999
    last_a_at_return_to_start_rise_time = -999999

    CT_array = [CT_start]
    a_array = [a_start]
    for tdx in range(1, len(tau_array)):
        tau = tau_array[tdx]
        a_current = a_array[-1]
        if tau >= 0 and tau < tau_switch:
            CT_current = CT_start
        elif tau >= tau_switch and tau < tau_return_to_start:
            CT_current = 0.8
        else:
            CT_current = CT_start

        corr = 1. - a_current
        LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_from_corr_and_chi(corr, chi)
        MM = actuator_coeff.get_MM_matrix()

        a_all = cas.DM(a_current)
        c_all = cas.DM(CT_current)
        c_numer = cas.DM(CT_current * actuator_dynamic_pressure * actuator_area)
        c_denom = cas.DM(actuator_dynamic_pressure * actuator_area)
        _, term_1_factor, term_2, term_3 = assemble_pitt_peters_style_residual(a_all, cas.DM(1.), t_star, t_star_num, t_star_den, MM[0, 0], LLinv[0, 0],
                                                                     LLinv_num[0, 0], LLinv_denom, c_all[0], c_numer[0],
                                                                     c_denom[0])
        da_dt_current = -1. * (term_2 + term_3) / term_1_factor

        timestep = (tau - tau_array[tdx -1]) * t_star
        a_found = a_current + da_dt_current * timestep

        if tau < tau_switch:
            last_a_before_switch = a_found
        if tau < tau_switch + observation_tau:
            last_a_at_switch_rise_time = a_found
        if tau < tau_return_to_start:
            last_a_before_return_to_start = a_found
        if tau < tau_return_to_start + observation_tau:
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

    def deltaV(a_local):
        return (a_start - a_local) / (a_switch - a_start)

    deltaV_array = [deltaV(a_local) for a_local in a_array]
    tau_array = np.array(cas.DM(tau_array))
    CT_array = np.array(cas.DM(CT_array))
    a_array = np.array(cas.DM(a_array))
    deltaV_array = np.array(cas.DM(deltaV_array))

    x_yu_array = [-0.008, 0.096, 0.2, 0.304, 0.408, 0.512, 0.616, 0.719, 0.823, 0.927, 1.031, 1.135, 1.239, 1.343,
                  1.447, 1.55, 1.654, 1.758, 1.862, 1.966, 2.07, 2.174, 2.278, 2.381, 2.485, 2.589, 2.693, 2.797, 2.901,
                  3.005, 3.109, 3.212, 3.316, 3.42, 3.524, 3.628, 3.732, 3.836, 3.939, 4.043, 4.147, 4.251, 4.355,
                  4.459, 4.563, 4.667, 4.77, 4.874, 4.978, 5.082, 5.186, 5.29, 5.394, 5.498, 5.601, 5.705, 5.809, 5.913,
                  6.017, 6.121, 6.225, 6.329, 6.432, 6.536, 6.64, 6.744, 6.848, 6.952, 7.056, 7.159, 7.263, 7.367,
                  7.471, 7.575, 7.679, 7.783, 7.887, 7.99, 8.094, 8.198, 8.302, 8.406, 8.51, 8.614, 8.718, 8.821, 8.925,
                  9.029, 9.133, 9.237, 9.341, 9.445, 9.549, 9.652, 9.756, 9.86, 9.964, 10.068, 10.172, 10.276, 10.379,
                  10.483, 10.587, 10.691, 10.795, 10.899, 11.003, 11.107, 11.21, 11.314, 11.418, 11.522, 11.626, 11.73,
                  11.834, 11.938, 12.041, 12.145, 12.249, 12.353, 12.457, 12.561, 12.665, 12.769, 12.872, 12.976, 13.08,
                  13.184, 13.288, 13.392, 13.496, 13.599, 13.703, 13.807, 13.911, 14.015, 14.119, 14.223, 14.327, 14.43,
                  14.534, 14.638, 14.742, 14.846, 14.95, 15.054, 15.158, 15.261, 15.365, 15.469, 15.573, 15.677, 15.781,
                  15.885, 15.989, 16.092, 16.196, 16.3, 16.404, 16.508, 16.612, 16.716, 16.82, 16.923, 17.027, 17.131,
                  17.235, 17.339, 17.443, 17.547, 17.65, 17.754, 17.858, 17.962, 18.066, 18.17, 18.274, 18.378, 18.481,
                  18.585, 18.689, 18.793, 18.897, 19.001, 19.105, 19.209, 19.312, 19.416, 19.52, 19.624, 19.728, 19.832,
                  19.936, 20.04, 20.143, 20.247, 20.351, 20.455, 20.559, 20.663, 20.767, 20.87, 20.974, 21.078, 21.182,
                  21.286, 21.39, 21.494, 21.598, 21.701, 21.805, 21.909, 22.013, 22.117, 22.221, 22.325, 22.429, 22.532,
                  22.636, 22.74, 22.844, 22.948, 23.052, 23.156, 23.26, 23.363, 23.467, 23.571, 23.675, 23.779, 23.883,
                  23.987, 24.09, 24.194, 24.298, 24.402, 24.506, 24.61, 24.714, 24.818, 24.921, 25.025, 25.129, 25.233,
                  25.337, 25.441, 25.545, 25.649, 25.752, 25.856, 25.96, 26.064, 26.168, 26.272, 26.376, 26.48, 26.583,
                  26.687, 26.791, 26.895, 26.999, 27.103, 27.207, 27.31, 27.414, 27.518, 27.622, 27.726, 27.83, 27.934,
                  28.038, 28.141, 28.245, 28.349, 28.453, 28.557, 28.661, 28.765, 28.869, 28.972, 29.076, 29.18, 29.284,
                  29.388, 29.492, 29.596, 29.7, 29.803, 29.907, 30.011, 30.115, 30.219, 30.323, 30.427, 30.53, 30.634,
                  30.738, 30.842, 30.946, 31.05, 31.154, 31.258, 31.361, 31.465, 31.569, 31.673, 31.777, 31.881, 31.985,
                  32.089, 32.192, 32.296, 32.4, 32.504, 32.608, 32.712, 32.816, 32.92, 33.023, 33.127, 33.231, 33.335,
                  33.439, 33.543, 33.647, 33.751, 33.854, 33.958, 34.062, 34.166, 34.27, 34.374, 34.478, 34.581, 34.685,
                  34.789, 34.893, 34.997, 35.101, 35.205, 35.309, 35.412, 35.516, 35.62, 35.724, 35.828, 35.932, 36.036,
                  36.14, 36.243, 36.347, 36.451, 36.555, 36.659, 36.763, 36.867, 36.971, 37.074, 37.178, 37.282, 37.386,
                  37.49, 37.594, 37.698, 37.801, 37.905, 38.009, 38.113, 38.217, 38.321, 38.425, 38.529, 38.632, 38.736,
                  38.84, 38.944, 39.048, 39.152, 39.256, 39.36, 39.463, 39.567, 39.671, 39.775, 39.879, 39.983, 40.087,
                  40.191, 40.294, 40.398, 40.502, 40.606, 40.71, 40.814, 40.918, 41.021, 41.125, 41.229, 41.333, 41.437,
                  41.541, 41.645, 41.749, 41.852, 41.956, 42.06, 42.164, 42.268, 42.372, 42.476, 42.58, 42.683, 42.787,
                  42.891, 42.995, 43.099, 43.203, 43.307, 43.411, 43.514, 43.618, 43.722, 43.826, 43.93, 44.034, 44.138,
                  44.241, 44.345, 44.449, 44.553, 44.657, 44.761, 44.865, 44.969, 45.072, 45.176, 45.28, 45.384, 45.488,
                  45.592, 45.696, 45.8, 45.903, 46.007, 46.111, 46.215, 46.319, 46.423, 46.527, 46.631, 46.734, 46.838,
                  46.942, 47.046, 47.15, 47.254, 47.358, 47.461, 47.565, 47.669, 47.773, 47.877, 47.981, 48.085, 48.189,
                  48.292, 48.396, 48.5, 48.604, 48.708, 48.812, 48.916, 49.02, 49.123, 49.227, 49.331, 49.435, 49.539,
                  49.643, 49.747, 49.851, 49.954, 50.058, 50.162, 50.266, 50.37, 50.474, 50.578, 50.681, 50.785, 50.889,
                  50.993, 51.097, 51.201, 51.305, 51.409, 51.512, 51.616, 51.72, 51.824, 51.928, 52.032, 52.136, 52.24,
                  52.343, 52.447, 52.551, 52.655, 52.759, 52.863, 52.967, 53.071, 53.174, 53.278, 53.382, 53.486, 53.59,
                  53.694, 53.798, 53.902, 54.005, 54.109, 54.213, 54.317, 54.421, 54.525, 54.629, 54.732, 54.836, 54.94,
                  55.044, 55.148, 55.252, 55.356, 55.46, 55.563, 55.667, 55.771, 55.875, 55.979, 56.083, 56.187, 56.291,
                  56.394, 56.498, 56.602, 56.706, 56.81, 56.914, 57.018, 57.122, 57.225, 57.329, 57.433, 57.537, 57.641,
                  57.745, 57.849, 57.952, 58.056, 58.16, 58.264, 58.368, 58.472, 58.576, 58.68, 58.783, 58.887, 58.991,
                  59.095, 59.199, 59.303, 59.407, 59.511, 59.614, 59.718, 59.822, 59.926, 60.03, 60.134, 60.238, 60.342,
                  60.445, 60.549, 60.653, 60.757, 60.861, 60.965, 61.069, 61.172, 61.276, 61.38, 61.484, 61.588, 61.692,
                  61.796, 61.9, 62.003, 62.107, 62.211, 62.315, 62.419, 62.523, 62.627, 62.731, 62.834, 62.938, 63.042,
                  63.146, 63.25, 63.354, 63.458, 63.562, 63.665, 63.769, 63.873, 63.977, 64.081, 64.185, 64.289, 64.392,
                  64.496, 64.6, 64.704, 64.808, 64.912, 65.016, 65.12, 65.223, 65.327, 65.431, 65.535, 65.639, 65.743,
                  65.847, 65.951, 66.054, 66.158, 66.262, 66.366, 66.47, 66.574, 66.678, 66.782, 66.885, 66.989, 67.093,
                  67.197, 67.301, 67.405, 67.509, 67.612, 67.716, 67.82, 67.924, 68.028, 68.132, 68.236, 68.34, 68.443,
                  68.547, 68.651, 68.755, 68.859, 68.963, 69.067, 69.171, 69.274, 69.378, 69.482, 69.586, 69.69, 69.794,
                  69.898, 70.002, 70.105, 70.209, 70.313, 70.417, 70.521, 70.625, 70.729, 70.833, 70.936, 71.04, 71.144,
                  71.248, 71.352, 71.456, 71.56, 71.663, 71.767, 71.871, 71.975, 72.079, 72.183, 72.287, 72.391, 72.494,
                  72.598, 72.702, 72.806, 72.91, 73.014, 73.118, 73.222, 73.325, 73.429, 73.533, 73.637, 73.741, 73.845,
                  73.949, 74.053, 74.156, 74.26, 74.364, 74.468, 74.572, 74.676, 74.78, 74.883, 74.987, 75.091, 75.195,
                  75.299, 75.403, 75.507, 75.611, 75.714, 75.818, 75.922, 76.026, 76.13, 76.234, 76.338, 76.442, 76.545,
                  76.649, 76.753, 76.857, 76.961, 77.065, 77.169, 77.273, 77.376, 77.48, 77.584, 77.688, 77.792, 77.896,
                  78, 78.103, 78.207, 78.311, 78.415, 78.519, 78.623, 78.727, 78.831, 78.934, 79.038, 79.142, 79.246,
                  79.35, 79.454, 79.558, 79.662, 79.765, 79.869, 79.973, 80.077, 80.181, 80.285, 80.389, 80.493, 80.596,
                  80.7, 80.804, 80.908, 81.012, 81.116, 81.22, 81.323, 81.427, 81.531, 81.635, 81.739, 81.843, 81.947,
                  82.051, 82.154, 82.258, 82.362, 82.466, 82.57, 82.674, 82.778, 82.882, 82.985, 83.089, 83.193, 83.297,
                  83.401, 83.505, 83.609, 83.713, 83.816, 83.92, 84.024, 84.128, 84.232, 84.336, 84.44, 84.543, 84.647,
                  84.751, 84.855, 84.959, 85.063, 85.167, 85.271, 85.374, 85.478, 85.582, 85.686, 85.79, 85.894, 85.998,
                  86.102, 86.205, 86.309, 86.413, 86.517, 86.621, 86.725, 86.829, 86.933, 87.036, 87.14, 87.244, 87.348,
                  87.452, 87.556, 87.66, 87.763, 87.867, 87.971, 88.075, 88.179, 88.283, 88.387, 88.491, 88.594, 88.698,
                  88.802, 88.906, 89.01, 89.114, 89.218, 89.322, 89.425, 89.529, 89.633, 89.737, 89.841, 89.945, 90.049,
                  90.153, 90.256, 90.36, 90.464, 90.568, 90.672, 90.776, 90.88, 90.984, 91.087, 91.191, 91.295, 91.399,
                  91.503, 91.607, 91.711, 91.814, 91.918, 92.022, 92.126, 92.23, 92.334, 92.438, 92.542, 92.645, 92.749,
                  92.853, 92.957, 93.061, 93.165, 93.269, 93.373, 93.476, 93.58, 93.684, 93.788, 93.892, 93.996, 94.1,
                  94.204, 94.307, 94.411, 94.515, 94.619, 94.723, 94.827, 94.931, 95.034, 95.138, 95.242, 95.346, 95.45,
                  95.554, 95.658, 95.762, 95.865, 95.969, 96.073, 96.177, 96.281, 96.385, 96.489, 96.593, 96.696, 96.8,
                  96.904, 97.008, 97.112, 97.216, 97.32, 97.424, 97.527, 97.631, 97.735, 97.839, 97.943, 98.047, 98.151,
                  98.254, 98.358, 98.462, 98.566, 98.67, 98.774, 98.878, 98.982, 99.085, 99.189, 99.293, 99.397, 99.501,
                  99.605, 99.709, 99.813, 99.916]
    qaxi_yu_array = [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    uaxi_yu_array = [-0.00013, -0.34809, -0.55629, -0.68451, -0.77011, -0.82974, -0.87227, -0.90379, -0.9278, -0.94632,
                     -0.96067, -0.97176, -0.98024, -0.98663, -0.99133, -0.99468, -0.99694, -0.99837, -0.99919, -0.99958,
                     -0.99975, -0.99985, -0.99995, -1.00002, -1.00009, -1.00014, -1.00018, -1.00022, -1.00025, -1.00027,
                     -1.0003, -1.00032, -1.00033, -1.00035, -1.00036, -1.00037, -1.00039, -1.0004, -1.00041, -1.00042,
                     -1.00042, -1.00043, -1.00044, -1.00045, -1.00046, -1.00046, -1.00047, -1.00048, -1.00048, -1.00049,
                     -1.00049, -1.0005, -1.00051, -1.00051, -1.00052, -1.00052, -1.00053, -1.00054, -1.00054, -1.00055,
                     -1.00055, -1.00056, -1.00056, -1.00057, -1.00058, -1.00058, -1.00059, -1.00059, -1.0006, -1.00061,
                     -1.00061, -1.00062, -1.00062, -1.00063, -1.00064, -1.00064, -1.00065, -1.00065, -1.00066, -1.00067,
                     -1.00067, -1.00068, -1.00068, -1.00069, -1.0007, -1.0007, -1.00071, -1.00072, -1.00072, -1.00073,
                     -1.00073, -1.00074, -1.00075, -1.00075, -1.00076, -1.00076, -1.00077, -1.00077, -1.00078, -1.00079,
                     -1.00079, -1.0008, -1.0008, -1.00081, -1.00082, -1.00082, -1.00083, -1.00083, -1.00084, -1.00084,
                     -1.00085, -1.00085, -1.00086, -1.00087, -1.00087, -1.00088, -1.00088, -1.00089, -1.00089, -1.0009,
                     -1.0009, -1.00091, -1.00091, -1.00092, -1.00092, -1.00093, -1.00093, -1.00094, -1.00094, -1.00095,
                     -1.00095, -1.00096, -1.00096, -1.00097, -1.00097, -1.00098, -1.00098, -1.00098, -1.00099, -1.00099,
                     -1.001, -1.001, -1.00101, -1.00101, -1.00102, -1.00102, -1.00102, -1.00103, -1.00103, -1.00104,
                     -1.00104, -1.00104, -1.00105, -1.00105, -1.00106, -1.00106, -1.00106, -1.00107, -1.00107, -1.00107,
                     -1.00108, -1.00108, -1.00108, -1.00109, -1.00109, -1.00109, -1.0011, -1.0011, -1.0011, -1.00111,
                     -1.00111, -1.00111, -1.00112, -1.00112, -1.00112, -1.00112, -1.00113, -1.00113, -1.00113, -1.00113,
                     -1.00114, -1.00114, -1.00114, -1.00114, -1.00115, -1.00115, -1.00115, -1.00115, -1.00115, -1.00116,
                     -1.00116, -1.00116, -1.00116, -1.00116, -1.00117, -1.00117, -1.00117, -1.00117, -1.00117, -1.00117,
                     -1.00117, -1.00118, -1.00118, -1.00118, -1.00118, -1.00118, -1.00118, -1.00118, -1.00118, -1.00118,
                     -1.00119, -1.00119, -1.00119, -1.00119, -1.00119, -1.00119, -1.00119, -1.00119, -1.00119, -1.00119,
                     -1.00119, -1.00119, -1.00119, -1.00119, -1.00119, -1.00119, -1.00119, -1.00119, -1.00119, -1.00119,
                     -1.00119, -1.00119, -1.00119, -1.00119, -1.00118, -1.00118, -1.00118, -1.00118, -1.00118, -1.00118,
                     -1.00118, -1.00118, -1.00118, -1.00117, -1.00117, -1.00117, -1.00117, -1.00117, -1.00117, -1.00117,
                     -1.00116, -1.00116, -1.00116, -1.00116, -1.00115, -1.00115, -1.00115, -1.00115, -1.00115, -1.00114,
                     -1.00114, -1.00114, -1.00113, -1.00113, -1.00113, -1.00113, -1.00112, -1.00112, -1.00112, -1.00111,
                     -1.00111, -1.00111, -1.0011, -1.0011, -1.0011, -1.00109, -1.00109, -1.00109, -1.00108, -1.00108,
                     -1.00107, -1.00107, -1.00107, -1.00106, -1.00106, -1.00105, -1.00105, -1.00105, -1.00104, -1.00104,
                     -1.00103, -1.00103, -1.00102, -1.00102, -1.00101, -1.00101, -1.00101, -1.001, -1.001, -1.00099,
                     -1.00099, -1.00098, -1.00098, -1.00097, -1.00097, -1.00096, -1.00095, -1.00095, -1.00094, -1.00094,
                     -1.00093, -1.00093, -1.00092, -1.00092, -1.00091, -1.00091, -1.0009, -1.00089, -1.00089, -1.00088,
                     -1.00088, -1.00087, -1.00086, -1.00086, -1.00085, -1.00085, -1.00084, -1.00083, -1.00083, -1.00082,
                     -1.00081, -1.00081, -1.0008, -1.0008, -1.00079, -1.00078, -1.00078, -1.00077, -1.00076, -1.00076,
                     -1.00075, -1.00074, -1.00074, -1.00073, -1.00072, -1.00071, -1.00071, -1.0007, -1.00069, -1.00069,
                     -1.00068, -1.00067, -1.00067, -1.00066, -1.00065, -1.00064, -1.00064, -1.00063, -1.00062, -1.00061,
                     -1.00061, -1.0006, -1.00059, -1.00058, -1.00058, -1.00057, -1.00056, -1.00055, -1.00055, -1.00054,
                     -1.00053, -1.00052, -1.00052, -1.00051, -1.0005, -1.00049, -1.00048, -1.00048, -1.00047, -1.00046,
                     -1.00045, -1.00044, -1.00044, -1.00043, -1.00042, -1.00041, -1.0004, -1.0004, -1.00039, -1.00038,
                     -1.00037, -1.00036, -1.00035, -1.00035, -1.00034, -1.00033, -1.00032, -1.00031, -1.0003, -1.0003,
                     -1.00029, -1.00028, -1.00027, -1.00026, -1.00025, -1.00024, -1.00024, -1.00023, -1.00022, -1.00021,
                     -1.0002, -1.00019, -1.00018, -1.00018, -1.00017, -1.00016, -1.00015, -1.00014, -1.00013, -1.00012,
                     -1.00011, -1.00011, -1.0001, -1.00009, -1.00008, -1.00007, -1.00006, -1.00005, -1.00004, -1.00003,
                     -1.00002, -1.00002, -1.00001, -1, -0.99999, -0.99998, -0.99997, -0.99996, -0.99995, -0.99994,
                     -0.99993, -0.99993, -0.99992, -0.99991, -0.9999, -0.99989, -0.99988, -0.99987, -0.99986, -0.99985,
                     -0.99984, -0.99983, -0.99983, -0.99982, -0.99981, -0.9998, -0.99979, -0.99978, -0.99977, -0.99976,
                     -0.99975, -0.99974, -0.99973, -0.99972, -0.99971, -0.99971, -0.9997, -0.99969, -0.99968, -0.99967,
                     -0.99966, -0.99965, -0.99964, -0.99963, -0.99962, -0.99961, -0.9996, -0.99959, -0.99959, -0.99958,
                     -0.99957, -0.99956, -0.85959, -0.66421, -0.4946, -0.35787, -0.25404, -0.17322, -0.10754, -0.05207,
                     -0.01718, -0.01024, -0.00709, -0.00519, -0.00391, -0.003, -0.00233, -0.00181, -0.00142, -0.00112,
                     -0.00088, -0.00069, -0.00055, -0.00044, -0.00036, -0.0003, -0.00025, -0.00023, -0.00021, -0.00021,
                     -0.00021, -0.00022, -0.00023, -0.00025, -0.00027, -0.0003, -0.00033, -0.00036, -0.00039, -0.00042,
                     -0.00045, -0.00048, -0.00051, -0.00054, -0.00057, -0.00059, -0.00062, -0.00064, -0.00067, -0.00069,
                     -0.00071, -0.00073, -0.00074, -0.00076, -0.00077, -0.00078, -0.0008, -0.00081, -0.00082, -0.00082,
                     -0.00083, -0.00084, -0.00085, -0.00085, -0.00085, -0.00086, -0.00086, -0.00086, -0.00086, -0.00087,
                     -0.00087, -0.00087, -0.00086, -0.00086, -0.00086, -0.00086, -0.00086, -0.00085, -0.00085, -0.00084,
                     -0.00084, -0.00083, -0.00083, -0.00082, -0.00082, -0.00081, -0.0008, -0.00079, -0.00079, -0.00078,
                     -0.00077, -0.00076, -0.00075, -0.00074, -0.00073, -0.00073, -0.00072, -0.00071, -0.00069, -0.00068,
                     -0.00067, -0.00066, -0.00065, -0.00064, -0.00063, -0.00062, -0.0006, -0.00059, -0.00058, -0.00057,
                     -0.00055, -0.00054, -0.00053, -0.00052, -0.0005, -0.00049, -0.00048, -0.00046, -0.00045, -0.00043,
                     -0.00042, -0.00041, -0.00039, -0.00038, -0.00036, -0.00035, -0.00034, -0.00032, -0.00031, -0.00029,
                     -0.00028, -0.00026, -0.00025, -0.00023, -0.00022, -0.0002, -0.00019, -0.00017, -0.00016, -0.00014,
                     -0.00013, -0.00011, -0.0001, -0.00008, -0.00006, -0.00005, -0.00003, -0.00002, -0.000003, 0.00001,
                     0.00003, 0.00004, 0.00006, 0.00008, 0.00009, 0.00011, 0.00012, 0.00014, 0.00015, 0.00017, 0.00018,
                     0.0002, 0.00022, 0.00023, 0.00025, 0.00026, 0.00028, 0.00029, 0.00031, 0.00032, 0.00034, 0.00035,
                     0.00037, 0.00039, 0.0004, 0.00042, 0.00043, 0.00045, 0.00046, 0.00048, 0.00049, 0.00051, 0.00052,
                     0.00054, 0.00055, 0.00056, 0.00058, 0.00059, 0.00061, 0.00062, 0.00064, 0.00065, 0.00067, 0.00068,
                     0.00069, 0.00071, 0.00072, 0.00074, 0.00075, 0.00076, 0.00078, 0.00079, 0.00081, 0.00082, 0.00083,
                     0.00085, 0.00086, 0.00087, 0.00088, 0.0009, 0.00091, 0.00092, 0.00094, 0.00095, 0.00096, 0.00097,
                     0.00098, 0.001, 0.00101, 0.00102, 0.00103, 0.00104, 0.00106, 0.00107, 0.00108, 0.00109, 0.0011,
                     0.00111, 0.00112, 0.00113, 0.00114, 0.00115, 0.00116, 0.00117, 0.00118, 0.00119, 0.0012, 0.00121,
                     0.00122, 0.00123, 0.00124, 0.00125, 0.00126, 0.00127, 0.00128, 0.00128, 0.00129, 0.0013, 0.00131,
                     0.00132, 0.00132, 0.00133, 0.00134, 0.00135, 0.00135, 0.00136, 0.00137, 0.00137, 0.00138, 0.00139,
                     0.00139, 0.0014, 0.0014, 0.00141, 0.00142, 0.00142, 0.00143, 0.00143, 0.00144, 0.00144, 0.00145,
                     0.00145, 0.00146, 0.00146, 0.00147, 0.00147, 0.00147, 0.00148, 0.00148, 0.00149, 0.00149, 0.00149,
                     0.0015, 0.0015, 0.0015, 0.00151, 0.00151, 0.00151, 0.00151, 0.00152, 0.00152, 0.00152, 0.00152,
                     0.00152, 0.00153, 0.00153, 0.00153, 0.00153, 0.00153, 0.00153, 0.00154, 0.00154, 0.00154, 0.00154,
                     0.00154, 0.00154, 0.00154, 0.00154, 0.00154, 0.00154, 0.00154, 0.00154, 0.00154, 0.00154, 0.00154,
                     0.00154, 0.00154, 0.00154, 0.00154, 0.00154, 0.00154, 0.00154, 0.00154, 0.00154, 0.00153, 0.00153,
                     0.00153, 0.00153, 0.00153, 0.00153, 0.00152, 0.00152, 0.00152, 0.00152, 0.00152, 0.00151, 0.00151,
                     0.00151, 0.00151, 0.0015, 0.0015, 0.0015, 0.0015, 0.00149, 0.00149, 0.00149, 0.00148, 0.00148,
                     0.00148, 0.00147, 0.00147, 0.00147, 0.00146, 0.00146, 0.00146, 0.00145, 0.00145, 0.00144, 0.00144,
                     0.00144, 0.00143, 0.00143, 0.00142, 0.00142, 0.00141, 0.00141, 0.0014, 0.0014, 0.00139, 0.00139,
                     0.00138, 0.00138, 0.00137, 0.00137, 0.00136, 0.00136, 0.00135, 0.00135, 0.00134, 0.00134, 0.00133,
                     0.00133, 0.00132, 0.00131, 0.00131, 0.0013, 0.0013, 0.00129, 0.00128, 0.00128, 0.00127, 0.00127,
                     0.00126, 0.00125, 0.00125, 0.00124, 0.00123, 0.00123, 0.00122, 0.00121, 0.00121, 0.0012, 0.00119,
                     0.00119, 0.00118, 0.00117, 0.00117, 0.00116, 0.00115, 0.00115, 0.00114, 0.00113, 0.00112, 0.00112,
                     0.00111, 0.0011, 0.0011, 0.00109, 0.00108, 0.00107, 0.00107, 0.00106, 0.00105, 0.00104, 0.00103,
                     0.00103, 0.00102, 0.00101, 0.001, 0.001, 0.00099, 0.00098, 0.00097, 0.00096, 0.00096, 0.00095,
                     0.00094, 0.00093, 0.00092, 0.00092, 0.00091, 0.0009, 0.00089, 0.00088, 0.00088, 0.00087, 0.00086,
                     0.00085, 0.00084, 0.00083, 0.00083, 0.00082, 0.00081, 0.0008, 0.00079, 0.00078, 0.00077, 0.00077,
                     0.00076, 0.00075, 0.00074, 0.00073, 0.00072, 0.00072, 0.00071, 0.0007, 0.00069, 0.00068, 0.00067,
                     0.00066, 0.00066, 0.00065, 0.00064, 0.00063, 0.00062, 0.00061, 0.0006, 0.00059, 0.00059, 0.00058,
                     0.00057, 0.00056, 0.00055]
    oye_yu_array = [0.00789, -0.11843, -0.24051, -0.35451, -0.45779, -0.54891, -0.62747, -0.69374, -0.74829, -0.79182,
                    -0.82605, -0.85307, -0.87465, -0.89215, -0.90654, -0.91852, -0.92861, -0.93721, -0.94458, -0.95097,
                    -0.95653, -0.9614, -0.96569, -0.96948, -0.97286, -0.97587, -0.97857, -0.98099, -0.98317, -0.98514,
                    -0.98692, -0.98853, -0.98998, -0.9913, -0.9925, -0.99357, -0.99455, -0.99543, -0.99623, -0.99695,
                    -0.99759, -0.99817, -0.99869, -0.99915, -0.99956, -0.99992, -1.00024, -1.00052, -1.00076, -1.00097,
                    -1.00114, -1.00129, -1.00141, -1.0015, -1.00158, -1.00163, -1.00166, -1.00168, -1.00168, -1.00167,
                    -1.00164, -1.0016, -1.00156, -1.0015, -1.00144, -1.00137, -1.00129, -1.00121, -1.00113, -1.00104,
                    -1.00095, -1.00086, -1.00076, -1.00067, -1.00058, -1.00049, -1.0004, -1.00031, -1.00023, -1.00015,
                    -1.00007, -1, -0.99993, -0.99986, -0.9998, -0.99975, -0.99969, -0.99964, -0.99959, -0.99955,
                    -0.99951, -0.99947, -0.99944, -0.99941, -0.99938, -0.99935, -0.99933, -0.9993, -0.99928, -0.99927,
                    -0.99925, -0.99924, -0.99923, -0.99922, -0.99921, -0.9992, -0.99919, -0.99919, -0.99919, -0.99919,
                    -0.99919, -0.99919, -0.99919, -0.99919, -0.9992, -0.9992, -0.99921, -0.99921, -0.99922, -0.99923,
                    -0.99924, -0.99925, -0.99926, -0.99927, -0.99928, -0.99929, -0.9993, -0.99931, -0.99933, -0.99934,
                    -0.99935, -0.99937, -0.99938, -0.9994, -0.99941, -0.99943, -0.99944, -0.99946, -0.99947, -0.99949,
                    -0.9995, -0.99952, -0.99953, -0.99955, -0.99957, -0.99958, -0.9996, -0.99962, -0.99963, -0.99965,
                    -0.99966, -0.99968, -0.9997, -0.99971, -0.99973, -0.99975, -0.99976, -0.99978, -0.99979, -0.99981,
                    -0.99982, -0.99984, -0.99986, -0.99987, -0.99989, -0.9999, -0.99992, -0.99993, -0.99995, -0.99996,
                    -0.99997, -0.99999, -1, -1.00002, -1.00003, -1.00004, -1.00006, -1.00007, -1.00008, -1.0001,
                    -1.00011, -1.00012, -1.00014, -1.00015, -1.00016, -1.00017, -1.00018, -1.00019, -1.00021, -1.00022,
                    -1.00023, -1.00024, -1.00025, -1.00026, -1.00027, -1.00028, -1.00029, -1.0003, -1.00031, -1.00032,
                    -1.00033, -1.00033, -1.00034, -1.00035, -1.00036, -1.00037, -1.00037, -1.00038, -1.00039, -1.0004,
                    -1.0004, -1.00041, -1.00042, -1.00042, -1.00043, -1.00043, -1.00044, -1.00044, -1.00045, -1.00046,
                    -1.00046, -1.00047, -1.00047, -1.00047, -1.00048, -1.00048, -1.00049, -1.00049, -1.00049, -1.0005,
                    -1.0005, -1.00051, -1.00051, -1.00051, -1.00051, -1.00052, -1.00052, -1.00052, -1.00053, -1.00053,
                    -1.00053, -1.00053, -1.00053, -1.00054, -1.00054, -1.00054, -1.00054, -1.00054, -1.00054, -1.00055,
                    -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055,
                    -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055,
                    -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00055, -1.00054, -1.00054,
                    -1.00054, -1.00054, -1.00054, -1.00054, -1.00054, -1.00053, -1.00053, -1.00053, -1.00053, -1.00053,
                    -1.00053, -1.00052, -1.00052, -1.00052, -1.00052, -1.00051, -1.00051, -1.00051, -1.00051, -1.0005,
                    -1.0005, -1.0005, -1.0005, -1.00049, -1.00049, -1.00049, -1.00049, -1.00048, -1.00048, -1.00048,
                    -1.00047, -1.00047, -1.00047, -1.00046, -1.00046, -1.00046, -1.00046, -1.00045, -1.00045, -1.00044,
                    -1.00044, -1.00044, -1.00043, -1.00043, -1.00043, -1.00042, -1.00042, -1.00042, -1.00041, -1.00041,
                    -1.0004, -1.0004, -1.0004, -1.00039, -1.00039, -1.00038, -1.00038, -1.00038, -1.00037, -1.00037,
                    -1.00036, -1.00036, -1.00035, -1.00035, -1.00035, -1.00034, -1.00034, -1.00033, -1.00033, -1.00032,
                    -1.00032, -1.00031, -1.00031, -1.0003, -1.0003, -1.00029, -1.00029, -1.00029, -1.00028, -1.00028,
                    -1.00027, -1.00027, -1.00026, -1.00026, -1.00025, -1.00025, -1.00024, -1.00024, -1.00023, -1.00023,
                    -1.00022, -1.00022, -1.00021, -1.0002, -1.0002, -1.00019, -1.00019, -1.00018, -1.00018, -1.00017,
                    -1.00017, -1.00016, -1.00016, -1.00015, -1.00015, -1.00014, -1.00013, -1.00013, -1.00012, -1.00012,
                    -1.00011, -1.00011, -1.0001, -1.0001, -1.00009, -1.00008, -1.00008, -1.00007, -1.00007, -1.00006,
                    -1.00006, -1.00005, -1.00004, -1.00004, -1.00003, -1.00003, -1.00002, -1.00001, -1.00001, -1, -1,
                    -0.99999, -0.99999, -0.99998, -0.99997, -0.99997, -0.99996, -0.99996, -0.99995, -0.99994, -0.99994,
                    -0.99993, -0.99992, -0.99992, -0.99991, -0.99991, -0.9999, -0.99989, -0.99989, -0.99988, -0.99988,
                    -0.99987, -0.99986, -0.99986, -0.99985, -0.99984, -0.99984, -0.99983, -0.99983, -0.99982, -0.99981,
                    -0.99981, -0.9998, -0.99979, -0.99979, -0.99978, -0.99978, -0.99977, -0.99976, -0.99976, -0.99975,
                    -0.99974, -0.99974, -0.99973, -0.99973, -0.99972, -0.99971, -0.99971, -0.9997, -0.99969, -0.99969,
                    -0.99968, -0.99967, -0.99967, -0.99966, -0.99966, -0.99965, -0.99964, -0.99964, -0.99963, -0.99962,
                    -0.99962, -0.99961, -0.9996, -0.9996, -0.99959, -0.99959, -0.99958, -0.99957, -0.99957, -0.99956,
                    -0.99955, -0.86131, -0.65736, -0.51684, -0.41812, -0.34592, -0.29117, -0.24764, -0.21188, -0.18189,
                    -0.15638, -0.13448, -0.11555, -0.09911, -0.0848, -0.07232, -0.06144, -0.05195, -0.04376, -0.03696,
                    -0.03146, -0.02705, -0.02349, -0.02058, -0.01818, -0.01618, -0.01448, -0.01304, -0.01179, -0.01071,
                    -0.00976, -0.00893, -0.0082, -0.00755, -0.00697, -0.00645, -0.00598, -0.00556, -0.00518, -0.00484,
                    -0.00453, -0.00425, -0.00399, -0.00376, -0.00354, -0.00335, -0.00317, -0.003, -0.00285, -0.00271,
                    -0.00258, -0.00247, -0.00236, -0.00226, -0.00216, -0.00207, -0.00199, -0.00192, -0.00185, -0.00178,
                    -0.00172, -0.00166, -0.00161, -0.00156, -0.00151, -0.00146, -0.00142, -0.00137, -0.00133, -0.0013,
                    -0.00126, -0.00122, -0.00119, -0.00115, -0.00112, -0.00109, -0.00106, -0.00103, -0.001, -0.00098,
                    -0.00095, -0.00093, -0.0009, -0.00088, -0.00086, -0.00083, -0.00081, -0.00079, -0.00077, -0.00075,
                    -0.00073, -0.00072, -0.0007, -0.00068, -0.00067, -0.00065, -0.00063, -0.00062, -0.00061, -0.00059,
                    -0.00058, -0.00057, -0.00055, -0.00054, -0.00053, -0.00052, -0.00051, -0.0005, -0.00049, -0.00048,
                    -0.00047, -0.00046, -0.00045, -0.00044, -0.00044, -0.00043, -0.00042, -0.00041, -0.00041, -0.0004,
                    -0.00039, -0.00039, -0.00038, -0.00038, -0.00037, -0.00037, -0.00036, -0.00036, -0.00035, -0.00035,
                    -0.00035, -0.00034, -0.00034, -0.00034, -0.00033, -0.00033, -0.00033, -0.00032, -0.00032, -0.00032,
                    -0.00032, -0.00032, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031,
                    -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031,
                    -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00031, -0.00032, -0.00032,
                    -0.00032, -0.00032, -0.00032, -0.00032, -0.00033, -0.00033, -0.00033, -0.00033, -0.00033, -0.00034,
                    -0.00034, -0.00034, -0.00034, -0.00034, -0.00035, -0.00035, -0.00035, -0.00035, -0.00036, -0.00036,
                    -0.00036, -0.00036, -0.00037, -0.00037, -0.00037, -0.00037, -0.00038, -0.00038, -0.00038, -0.00038,
                    -0.00039, -0.00039, -0.00039, -0.0004, -0.0004, -0.0004, -0.0004, -0.00041, -0.00041, -0.00041,
                    -0.00041, -0.00042, -0.00042, -0.00042, -0.00043, -0.00043, -0.00043, -0.00043, -0.00044, -0.00044,
                    -0.00044, -0.00044, -0.00045, -0.00045, -0.00045, -0.00045, -0.00046, -0.00046, -0.00046, -0.00046,
                    -0.00047, -0.00047, -0.00047, -0.00047, -0.00048, -0.00048, -0.00048, -0.00048, -0.00048, -0.00049,
                    -0.00049, -0.00049, -0.00049, -0.00049, -0.0005, -0.0005, -0.0005, -0.0005, -0.0005, -0.0005,
                    -0.00051, -0.00051, -0.00051, -0.00051, -0.00051, -0.00051, -0.00051, -0.00051, -0.00051, -0.00052,
                    -0.00052, -0.00052, -0.00052, -0.00052, -0.00052, -0.00052, -0.00052, -0.00052, -0.00052, -0.00052,
                    -0.00052, -0.00052, -0.00052, -0.00052, -0.00052, -0.00052, -0.00052, -0.00052, -0.00052, -0.00051,
                    -0.00051, -0.00051, -0.00051, -0.00051, -0.00051, -0.00051, -0.00051, -0.00051, -0.0005, -0.0005,
                    -0.0005, -0.0005, -0.0005, -0.0005, -0.0005, -0.00049, -0.00049, -0.00049, -0.00049, -0.00049,
                    -0.00048, -0.00048, -0.00048, -0.00048, -0.00047, -0.00047, -0.00047, -0.00047, -0.00046, -0.00046,
                    -0.00046, -0.00046, -0.00045, -0.00045, -0.00045, -0.00044, -0.00044, -0.00044, -0.00043, -0.00043,
                    -0.00043, -0.00042, -0.00042, -0.00042, -0.00041, -0.00041, -0.00041, -0.0004, -0.0004, -0.00039,
                    -0.00039, -0.00039, -0.00038, -0.00038, -0.00037, -0.00037, -0.00036, -0.00036, -0.00036, -0.00035,
                    -0.00035, -0.00034, -0.00034, -0.00033, -0.00033, -0.00032, -0.00032, -0.00031, -0.00031, -0.0003,
                    -0.0003, -0.00029, -0.00029, -0.00028, -0.00028, -0.00027, -0.00027, -0.00026, -0.00026, -0.00025,
                    -0.00025, -0.00024, -0.00024, -0.00023, -0.00023, -0.00022, -0.00021, -0.00021, -0.0002, -0.0002,
                    -0.00019, -0.00018, -0.00018, -0.00017, -0.00017, -0.00016, -0.00015, -0.00015, -0.00014, -0.00014,
                    -0.00013, -0.00012, -0.00012, -0.00011, -0.0001, -0.0001, -0.00009, -0.00009, -0.00008, -0.00007,
                    -0.00007, -0.00006, -0.00005, -0.00005, -0.00004, -0.00003, -0.00003, -0.00002, -0.00001, -0.000005,
                    0.000002, 0.000009, 0.00002, 0.00002, 0.00003, 0.00004, 0.00004, 0.00005, 0.00006, 0.00007, 0.00007,
                    0.00008, 0.00009, 0.00009, 0.0001, 0.00011, 0.00012, 0.00012, 0.00013, 0.00014, 0.00015, 0.00015,
                    0.00016, 0.00017, 0.00018, 0.00018, 0.00019, 0.0002, 0.00021, 0.00021, 0.00022, 0.00023, 0.00024,
                    0.00024, 0.00025, 0.00026, 0.00027, 0.00028, 0.00028, 0.00029, 0.0003, 0.00031, 0.00031, 0.00032,
                    0.00033, 0.00034, 0.00035, 0.00035, 0.00036, 0.00037, 0.00038, 0.00039, 0.00039, 0.0004, 0.00041,
                    0.00042, 0.00043, 0.00043, 0.00044, 0.00045, 0.00046, 0.00047, 0.00047, 0.00048, 0.00049, 0.0005,
                    0.00051, 0.00051, 0.00052, 0.00053, 0.00054, 0.00055, 0.00055, 0.00056, 0.00057, 0.00058, 0.00059,
                    0.00059, 0.0006, 0.00061, 0.00062, 0.00063]
    ecn_yu_array = [0.00704, -0.10738, -0.22011, -0.32937, -0.43336, -0.53029, -0.61838, -0.69583, -0.75994, -0.80793,
                    -0.84253, -0.86789, -0.88707, -0.90202, -0.914, -0.92381, -0.93198, -0.9389, -0.94484, -0.94998,
                    -0.95449, -0.95846, -0.962, -0.96517, -0.96803, -0.97061, -0.97294, -0.97507, -0.97701, -0.97878,
                    -0.9804, -0.98189, -0.98326, -0.98452, -0.98568, -0.98675, -0.98774, -0.98866, -0.9895, -0.99029,
                    -0.99102, -0.99169, -0.99232, -0.9929, -0.99344, -0.99394, -0.99441, -0.99484, -0.99524, -0.99562,
                    -0.99597, -0.99629, -0.99659, -0.99688, -0.99714, -0.99738, -0.9976, -0.99781, -0.99801, -0.99819,
                    -0.99836, -0.99852, -0.99866, -0.9988, -0.99892, -0.99904, -0.99915, -0.99925, -0.99934, -0.99943,
                    -0.99951, -0.99959, -0.99966, -0.99972, -0.99979, -0.99984, -0.9999, -0.99995, -1, -1.00004,
                    -1.00009, -1.00013, -1.00017, -1.0002, -1.00024, -1.00027, -1.00031, -1.00034, -1.00037, -1.0004,
                    -1.00043, -1.00046, -1.00049, -1.00052, -1.00055, -1.00057, -1.0006, -1.00062, -1.00064, -1.00067,
                    -1.00069, -1.00071, -1.00073, -1.00075, -1.00077, -1.00079, -1.00081, -1.00082, -1.00084, -1.00086,
                    -1.00087, -1.00089, -1.0009, -1.00092, -1.00093, -1.00095, -1.00096, -1.00097, -1.00098, -1.001,
                    -1.00101, -1.00102, -1.00103, -1.00104, -1.00105, -1.00106, -1.00107, -1.00108, -1.00108, -1.00109,
                    -1.0011, -1.00111, -1.00111, -1.00112, -1.00113, -1.00113, -1.00114, -1.00114, -1.00115, -1.00115,
                    -1.00116, -1.00116, -1.00117, -1.00117, -1.00118, -1.00118, -1.00118, -1.00119, -1.00119, -1.00119,
                    -1.00119, -1.0012, -1.0012, -1.0012, -1.0012, -1.0012, -1.0012, -1.0012, -1.0012, -1.00121,
                    -1.00121, -1.00121, -1.00121, -1.00121, -1.00121, -1.00121, -1.00121, -1.00121, -1.0012, -1.0012,
                    -1.0012, -1.0012, -1.0012, -1.0012, -1.0012, -1.0012, -1.00119, -1.00119, -1.00119, -1.00119,
                    -1.00119, -1.00118, -1.00118, -1.00118, -1.00118, -1.00117, -1.00117, -1.00117, -1.00117, -1.00116,
                    -1.00116, -1.00116, -1.00115, -1.00115, -1.00115, -1.00114, -1.00114, -1.00114, -1.00113, -1.00113,
                    -1.00113, -1.00112, -1.00112, -1.00111, -1.00111, -1.00111, -1.0011, -1.0011, -1.00109, -1.00109,
                    -1.00108, -1.00108, -1.00108, -1.00107, -1.00107, -1.00106, -1.00106, -1.00105, -1.00105, -1.00104,
                    -1.00104, -1.00103, -1.00103, -1.00102, -1.00102, -1.00101, -1.00101, -1.001, -1.001, -1.00099,
                    -1.00099, -1.00098, -1.00098, -1.00097, -1.00097, -1.00096, -1.00096, -1.00095, -1.00095, -1.00094,
                    -1.00094, -1.00093, -1.00093, -1.00092, -1.00091, -1.00091, -1.0009, -1.0009, -1.00089, -1.00089,
                    -1.00088, -1.00088, -1.00087, -1.00087, -1.00086, -1.00085, -1.00085, -1.00084, -1.00084, -1.00083,
                    -1.00083, -1.00082, -1.00081, -1.00081, -1.0008, -1.0008, -1.00079, -1.00079, -1.00078, -1.00077,
                    -1.00077, -1.00076, -1.00076, -1.00075, -1.00075, -1.00074, -1.00073, -1.00073, -1.00072, -1.00072,
                    -1.00071, -1.0007, -1.0007, -1.00069, -1.00069, -1.00068, -1.00067, -1.00067, -1.00066, -1.00066,
                    -1.00065, -1.00064, -1.00064, -1.00063, -1.00063, -1.00062, -1.00061, -1.00061, -1.0006, -1.0006,
                    -1.00059, -1.00058, -1.00058, -1.00057, -1.00057, -1.00056, -1.00055, -1.00055, -1.00054, -1.00053,
                    -1.00053, -1.00052, -1.00052, -1.00051, -1.0005, -1.0005, -1.00049, -1.00048, -1.00048, -1.00047,
                    -1.00047, -1.00046, -1.00045, -1.00045, -1.00044, -1.00043, -1.00043, -1.00042, -1.00042, -1.00041,
                    -1.0004, -1.0004, -1.00039, -1.00038, -1.00038, -1.00037, -1.00036, -1.00036, -1.00035, -1.00035,
                    -1.00034, -1.00033, -1.00033, -1.00032, -1.00031, -1.00031, -1.0003, -1.00029, -1.00029, -1.00028,
                    -1.00027, -1.00027, -1.00026, -1.00026, -1.00025, -1.00024, -1.00024, -1.00023, -1.00022, -1.00022,
                    -1.00021, -1.0002, -1.0002, -1.00019, -1.00018, -1.00018, -1.00017, -1.00016, -1.00016, -1.00015,
                    -1.00014, -1.00014, -1.00013, -1.00012, -1.00012, -1.00011, -1.0001, -1.0001, -1.00009, -1.00008,
                    -1.00008, -1.00007, -1.00006, -1.00006, -1.00005, -1.00005, -1.00004, -1.00003, -1.00003, -1.00002,
                    -1.00001, -1.00001, -1, -0.99999, -0.99999, -0.99998, -0.99997, -0.99997, -0.99996, -0.99995,
                    -0.99995, -0.99994, -0.99993, -0.99993, -0.99992, -0.99991, -0.99991, -0.9999, -0.99989, -0.99988,
                    -0.99988, -0.99987, -0.99986, -0.99986, -0.99985, -0.99984, -0.99984, -0.99983, -0.99982, -0.99982,
                    -0.99981, -0.9998, -0.9998, -0.99979, -0.99978, -0.99978, -0.99977, -0.99976, -0.99976, -0.99975,
                    -0.99974, -0.99974, -0.99973, -0.99972, -0.99972, -0.99971, -0.9997, -0.9997, -0.99969, -0.99968,
                    -0.99968, -0.99967, -0.99966, -0.99966, -0.99965, -0.99964, -0.99964, -0.99963, -0.99962, -0.99961,
                    -0.99961, -0.9996, -0.99959, -0.99959, -0.99958, -0.99957, -0.99957, -0.99956, -0.99955, -0.99955,
                    -0.99954, -0.99953, -0.99953, -0.99952, -0.99951, -0.99951, -0.9995, -0.99949, -0.99949, -0.99948,
                    -0.99947, -0.99947, -0.99946, -0.99945, -0.99945, -0.99944, -0.99943, -0.99942, -0.99942, -0.99941,
                    -0.9994, -0.9994, -0.93579, -0.80845, -0.68859, -0.57976, -0.48399, -0.40197, -0.33349, -0.27773,
                    -0.23256, -0.1954, -0.1645, -0.13862, -0.11681, -0.09831, -0.08249, -0.06871, -0.05635, -0.0451,
                    -0.03594, -0.02907, -0.02395, -0.02003, -0.01693, -0.01443, -0.01236, -0.01062, -0.00913, -0.00785,
                    -0.00674, -0.00577, -0.00491, -0.00415, -0.00347, -0.00286, -0.00232, -0.00184, -0.0014, -0.001,
                    -0.00065, -0.00033, -0.00004, 0.00023, 0.00046, 0.00068, 0.00087, 0.00104, 0.0012, 0.00134, 0.00146,
                    0.00158, 0.00168, 0.00177, 0.00185, 0.00192, 0.00198, 0.00203, 0.00208, 0.00212, 0.00215, 0.00218,
                    0.0022, 0.00222, 0.00223, 0.00224, 0.00225, 0.00225, 0.00224, 0.00224, 0.00223, 0.00222, 0.00221,
                    0.00219, 0.00217, 0.00215, 0.00213, 0.00211, 0.00208, 0.00205, 0.00202, 0.00199, 0.00196, 0.00193,
                    0.0019, 0.00186, 0.00183, 0.00179, 0.00176, 0.00172, 0.00168, 0.00165, 0.00161, 0.00157, 0.00153,
                    0.00149, 0.00145, 0.00141, 0.00137, 0.00133, 0.00129, 0.00125, 0.00121, 0.00117, 0.00113, 0.00108,
                    0.00104, 0.001, 0.00096, 0.00092, 0.00088, 0.00084, 0.0008, 0.00076, 0.00072, 0.00068, 0.00064,
                    0.0006, 0.00056, 0.00052, 0.00048, 0.00045, 0.00041, 0.00037, 0.00033, 0.0003, 0.00026, 0.00022,
                    0.00019, 0.00015, 0.00012, 0.00008, 0.00005, 0.00001, -0.00002, -0.00006, -0.00009, -0.00012,
                    -0.00016, -0.00019, -0.00022, -0.00025, -0.00028, -0.00031, -0.00034, -0.00037, -0.0004, -0.00043,
                    -0.00046, -0.00049, -0.00051, -0.00054, -0.00057, -0.00059, -0.00062, -0.00064, -0.00067, -0.00069,
                    -0.00072, -0.00074, -0.00076, -0.00079, -0.00081, -0.00083, -0.00085, -0.00087, -0.00089, -0.00091,
                    -0.00093, -0.00095, -0.00097, -0.00099, -0.00101, -0.00103, -0.00104, -0.00106, -0.00108, -0.00109,
                    -0.00111, -0.00112, -0.00114, -0.00115, -0.00117, -0.00118, -0.00119, -0.00121, -0.00122, -0.00123,
                    -0.00125, -0.00126, -0.00127, -0.00128, -0.00129, -0.0013, -0.00131, -0.00132, -0.00133, -0.00134,
                    -0.00135, -0.00136, -0.00137, -0.00138, -0.00139, -0.0014, -0.0014, -0.00141, -0.00142, -0.00143,
                    -0.00143, -0.00144, -0.00145, -0.00145, -0.00146, -0.00146, -0.00147, -0.00147, -0.00148, -0.00148,
                    -0.00149, -0.00149, -0.0015, -0.0015, -0.00151, -0.00151, -0.00151, -0.00152, -0.00152, -0.00152,
                    -0.00153, -0.00153, -0.00153, -0.00153, -0.00154, -0.00154, -0.00154, -0.00154, -0.00154, -0.00154,
                    -0.00154, -0.00155, -0.00155, -0.00155, -0.00155, -0.00155, -0.00155, -0.00155, -0.00155, -0.00155,
                    -0.00155, -0.00155, -0.00155, -0.00155, -0.00154, -0.00154, -0.00154, -0.00154, -0.00154, -0.00154,
                    -0.00154, -0.00153, -0.00153, -0.00153, -0.00153, -0.00153, -0.00152, -0.00152, -0.00152, -0.00152,
                    -0.00151, -0.00151, -0.00151, -0.0015, -0.0015, -0.0015, -0.00149, -0.00149, -0.00148, -0.00148,
                    -0.00148, -0.00147, -0.00147, -0.00146, -0.00146, -0.00146, -0.00145, -0.00145, -0.00144, -0.00144,
                    -0.00143, -0.00143, -0.00142, -0.00142, -0.00141, -0.0014, -0.0014, -0.00139, -0.00139, -0.00138,
                    -0.00138, -0.00137, -0.00136, -0.00136, -0.00135, -0.00135, -0.00134, -0.00133, -0.00133, -0.00132,
                    -0.00131, -0.00131, -0.0013, -0.00129, -0.00128, -0.00128, -0.00127, -0.00126, -0.00126, -0.00125,
                    -0.00124, -0.00123, -0.00123, -0.00122, -0.00121, -0.0012, -0.0012, -0.00119, -0.00118, -0.00117,
                    -0.00116, -0.00115, -0.00115, -0.00114, -0.00113, -0.00112, -0.00111, -0.0011, -0.0011, -0.00109,
                    -0.00108, -0.00107, -0.00106, -0.00105, -0.00104, -0.00103, -0.00103, -0.00102, -0.00101, -0.001,
                    -0.00099, -0.00098, -0.00097, -0.00096, -0.00095, -0.00094, -0.00093, -0.00092, -0.00091, -0.0009,
                    -0.00089, -0.00088, -0.00087, -0.00086, -0.00085, -0.00084, -0.00083, -0.00082, -0.00081, -0.0008,
                    -0.00079, -0.00078, -0.00077, -0.00076, -0.00075, -0.00074, -0.00073, -0.00072, -0.00071, -0.0007,
                    -0.00069, -0.00068, -0.00067, -0.00066, -0.00065, -0.00064, -0.00062, -0.00061, -0.0006, -0.00059,
                    -0.00058, -0.00057, -0.00056, -0.00055, -0.00054, -0.00053, -0.00051, -0.0005, -0.00049, -0.00048,
                    -0.00047, -0.00046, -0.00045, -0.00044, -0.00042, -0.00041, -0.0004, -0.00039, -0.00038, -0.00037,
                    -0.00036, -0.00034, -0.00033, -0.00032, -0.00031, -0.0003, -0.00029, -0.00027, -0.00026, -0.00025,
                    -0.00024, -0.00023, -0.00022, -0.0002, -0.00019, -0.00018, -0.00017, -0.00016, -0.00014, -0.00013,
                    -0.00012, -0.00011, -0.0001, -0.00008, -0.00007, -0.00006, -0.00005, -0.00004, -0.00002, -0.00001,
                    0.00000041583, 0.00001, 0.00002, 0.00004, 0.00005, 0.00006, 0.00007, 0.00009, 0.0001, 0.00011,
                    0.00012, 0.00013, 0.00015, 0.00016, 0.00017, 0.00018, 0.0002, 0.00021, 0.00022, 0.00023, 0.00025,
                    0.00026, 0.00027, 0.00028, 0.0003, 0.00031, 0.00032, 0.00033, 0.00034, 0.00036, 0.00037, 0.00038,
                    0.00039, 0.00041, 0.00042, 0.00043, 0.00044, 0.00046, 0.00047, 0.00048, 0.00049, 0.00051, 0.00052,
                    0.00053, 0.00054]
    vtn_yu_array = [0.01372, -0.34992, -0.49695, -0.58669, -0.65091, -0.70029, -0.73982, -0.77227, -0.79936, -0.8222,
                    -0.84165, -0.85841, -0.87302, -0.88585, -0.89718, -0.90726, -0.91624, -0.92428, -0.93148, -0.93796,
                    -0.94378, -0.94903, -0.95376, -0.95803, -0.96188, -0.96534, -0.96847, -0.97129, -0.97382, -0.9761,
                    -0.97814, -0.97997, -0.98161, -0.98307, -0.98437, -0.98553, -0.98656, -0.98746, -0.98826, -0.98896,
                    -0.98958, -0.99011, -0.99058, -0.99099, -0.99134, -0.99165, -0.99191, -0.99215, -0.99235, -0.99253,
                    -0.9927, -0.99286, -0.993, -0.99315, -0.99329, -0.99343, -0.99357, -0.99371, -0.99384, -0.99397,
                    -0.9941, -0.99423, -0.99435, -0.99448, -0.9946, -0.99472, -0.99484, -0.99495, -0.99507, -0.99518,
                    -0.99529, -0.9954, -0.99551, -0.99562, -0.99572, -0.99583, -0.99593, -0.99603, -0.99613, -0.99623,
                    -0.99632, -0.99642, -0.99651, -0.99661, -0.9967, -0.99679, -0.99687, -0.99696, -0.99705, -0.99713,
                    -0.99722, -0.9973, -0.99738, -0.99746, -0.99754, -0.99761, -0.99769, -0.99777, -0.99784, -0.99791,
                    -0.99798, -0.99805, -0.99812, -0.99819, -0.99826, -0.99832, -0.99838, -0.99845, -0.99851, -0.99857,
                    -0.99863, -0.99869, -0.99874, -0.9988, -0.99885, -0.9989, -0.99896, -0.99901, -0.99905, -0.9991,
                    -0.99915, -0.99919, -0.99923, -0.99928, -0.99932, -0.99935, -0.99939, -0.99943, -0.99946, -0.99949,
                    -0.99952, -0.99955, -0.99958, -0.9996, -0.99963, -0.99965, -0.99967, -0.99969, -0.99971, -0.99973,
                    -0.99975, -0.99976, -0.99978, -0.99979, -0.9998, -0.99982, -0.99983, -0.99984, -0.99985, -0.99985,
                    -0.99986, -0.99987, -0.99988, -0.99988, -0.99989, -0.99989, -0.9999, -0.9999, -0.9999, -0.99991,
                    -0.99991, -0.99991, -0.99991, -0.99991, -0.99991, -0.99992, -0.99992, -0.99992, -0.99992, -0.99992,
                    -0.99992, -0.99992, -0.99992, -0.99992, -0.99991, -0.99991, -0.99991, -0.99991, -0.99991, -0.99991,
                    -0.99991, -0.99991, -0.9999, -0.9999, -0.9999, -0.9999, -0.9999, -0.99989, -0.99989, -0.99989,
                    -0.99989, -0.99988, -0.99988, -0.99988, -0.99988, -0.99988, -0.99987, -0.99987, -0.99987, -0.99987,
                    -0.99986, -0.99986, -0.99986, -0.99986, -0.99985, -0.99985, -0.99985, -0.99985, -0.99984, -0.99984,
                    -0.99984, -0.99984, -0.99983, -0.99983, -0.99983, -0.99983, -0.99982, -0.99982, -0.99982, -0.99982,
                    -0.99982, -0.99981, -0.99981, -0.99981, -0.99981, -0.9998, -0.9998, -0.9998, -0.9998, -0.9998,
                    -0.99979, -0.99979, -0.99979, -0.99979, -0.99979, -0.99978, -0.99978, -0.99978, -0.99978, -0.99978,
                    -0.99977, -0.99977, -0.99977, -0.99977, -0.99977, -0.99976, -0.99976, -0.99976, -0.99976, -0.99976,
                    -0.99975, -0.99975, -0.99975, -0.99975, -0.99975, -0.99974, -0.99974, -0.99974, -0.99974, -0.99974,
                    -0.99974, -0.99973, -0.99973, -0.99973, -0.99973, -0.99973, -0.99972, -0.99972, -0.99972, -0.99972,
                    -0.99972, -0.99972, -0.99971, -0.99971, -0.99971, -0.99971, -0.99971, -0.99971, -0.9997, -0.9997,
                    -0.9997, -0.9997, -0.9997, -0.9997, -0.99969, -0.99969, -0.99969, -0.99969, -0.99969, -0.99969,
                    -0.99968, -0.99968, -0.99968, -0.99968, -0.99968, -0.99968, -0.99967, -0.99967, -0.99967, -0.99967,
                    -0.99967, -0.99967, -0.99966, -0.99966, -0.99966, -0.99966, -0.99966, -0.99966, -0.99965, -0.99965,
                    -0.99965, -0.99965, -0.99965, -0.99965, -0.99965, -0.99964, -0.99964, -0.99964, -0.99964, -0.99964,
                    -0.99964, -0.99963, -0.99963, -0.99963, -0.99963, -0.99963, -0.99963, -0.99963, -0.99962, -0.99962,
                    -0.99962, -0.99962, -0.99962, -0.99962, -0.99961, -0.99961, -0.99961, -0.99961, -0.99961, -0.99961,
                    -0.99961, -0.9996, -0.9996, -0.9996, -0.9996, -0.9996, -0.9996, -0.9996, -0.99959, -0.99959,
                    -0.99959, -0.99959, -0.99959, -0.99959, -0.99959, -0.99958, -0.99958, -0.99958, -0.99958, -0.99958,
                    -0.99958, -0.99958, -0.99957, -0.99957, -0.99957, -0.99957, -0.99957, -0.99957, -0.99956, -0.99956,
                    -0.99956, -0.99956, -0.99956, -0.99956, -0.99956, -0.99955, -0.99955, -0.99955, -0.99955, -0.99955,
                    -0.99955, -0.99955, -0.99955, -0.99954, -0.99954, -0.99954, -0.99954, -0.99954, -0.99954, -0.99954,
                    -0.99953, -0.99953, -0.99953, -0.99953, -0.99953, -0.99953, -0.99953, -0.99952, -0.99952, -0.99952,
                    -0.99952, -0.99952, -0.99952, -0.99952, -0.99951, -0.99951, -0.99951, -0.99951, -0.99951, -0.99951,
                    -0.99951, -0.9995, -0.9995, -0.9995, -0.9995, -0.9995, -0.9995, -0.9995, -0.99949, -0.99949,
                    -0.99949, -0.99949, -0.99949, -0.99949, -0.99949, -0.99949, -0.99948, -0.99948, -0.99948, -0.99948,
                    -0.99948, -0.99948, -0.99948, -0.99947, -0.99947, -0.99947, -0.99947, -0.99947, -0.99947, -0.99947,
                    -0.99946, -0.99946, -0.99946, -0.99946, -0.99946, -0.99946, -0.99946, -0.99946, -0.99945, -0.99945,
                    -0.99945, -0.99945, -0.99945, -0.99945, -0.99945, -0.99944, -0.99944, -0.99944, -0.99944, -0.99944,
                    -0.99944, -0.99944, -0.99944, -0.99943, -0.99943, -0.99943, -0.99943, -0.99943, -0.99943, -0.99943,
                    -0.99942, -0.99942, -0.99942, -0.99942, -0.99942, -0.99942, -0.99942, -0.99941, -0.99941, -0.99941,
                    -0.99941, -0.99941, -0.94242, -0.82706, -0.71442, -0.60632, -0.50467, -0.41139, -0.32852, -0.25818,
                    -0.20236, -0.16062, -0.13085, -0.11087, -0.0984, -0.09101, -0.08608, -0.08077, -0.07382, -0.06677,
                    -0.06046, -0.05497, -0.05019, -0.046, -0.04229, -0.03899, -0.03603, -0.03336, -0.03095, -0.02875,
                    -0.02675, -0.02492, -0.02325, -0.02173, -0.02033, -0.01904, -0.01787, -0.01679, -0.01581, -0.0149,
                    -0.01406, -0.01329, -0.01257, -0.01191, -0.0113, -0.01073, -0.0102, -0.0097, -0.00924, -0.00881,
                    -0.0084, -0.00802, -0.00767, -0.00733, -0.00702, -0.00672, -0.00644, -0.00617, -0.00593, -0.00569,
                    -0.00547, -0.00525, -0.00505, -0.00486, -0.00468, -0.00451, -0.00435, -0.00419, -0.00404, -0.0039,
                    -0.00377, -0.00364, -0.00352, -0.0034, -0.00329, -0.00319, -0.00309, -0.00299, -0.0029, -0.00281,
                    -0.00272, -0.00264, -0.00257, -0.00249, -0.00242, -0.00235, -0.00229, -0.00223, -0.00217, -0.00211,
                    -0.00206, -0.002, -0.00195, -0.00191, -0.00186, -0.00181, -0.00177, -0.00173, -0.00169, -0.00165,
                    -0.00162, -0.00158, -0.00155, -0.00152, -0.00149, -0.00146, -0.00143, -0.0014, -0.00137, -0.00135,
                    -0.00132, -0.0013, -0.00128, -0.00125, -0.00123, -0.00121, -0.00119, -0.00117, -0.00115, -0.00114,
                    -0.00112, -0.0011, -0.00109, -0.00107, -0.00106, -0.00104, -0.00103, -0.00101, -0.001, -0.00099,
                    -0.00098, -0.00096, -0.00095, -0.00094, -0.00093, -0.00092, -0.00091, -0.0009, -0.00089, -0.00088,
                    -0.00087, -0.00086, -0.00085, -0.00084, -0.00083, -0.00082, -0.00081, -0.0008, -0.00079, -0.00079,
                    -0.00078, -0.00077, -0.00076, -0.00075, -0.00074, -0.00074, -0.00073, -0.00072, -0.00071, -0.00071,
                    -0.0007, -0.00069, -0.00068, -0.00068, -0.00067, -0.00066, -0.00066, -0.00065, -0.00064, -0.00063,
                    -0.00063, -0.00062, -0.00061, -0.00061, -0.0006, -0.0006, -0.00059, -0.00058, -0.00058, -0.00057,
                    -0.00056, -0.00056, -0.00055, -0.00055, -0.00054, -0.00053, -0.00053, -0.00052, -0.00052, -0.00051,
                    -0.00051, -0.0005, -0.00049, -0.00049, -0.00048, -0.00048, -0.00047, -0.00047, -0.00046, -0.00046,
                    -0.00045, -0.00045, -0.00044, -0.00044, -0.00043, -0.00043, -0.00042, -0.00042, -0.00041, -0.00041,
                    -0.0004, -0.0004, -0.00039, -0.00039, -0.00038, -0.00038, -0.00037, -0.00037, -0.00036, -0.00036,
                    -0.00036, -0.00035, -0.00035, -0.00034, -0.00034, -0.00033, -0.00033, -0.00032, -0.00032, -0.00032,
                    -0.00031, -0.00031, -0.0003, -0.0003, -0.0003, -0.00029, -0.00029, -0.00028, -0.00028, -0.00028,
                    -0.00027, -0.00027, -0.00026, -0.00026, -0.00026, -0.00025, -0.00025, -0.00025, -0.00024, -0.00024,
                    -0.00023, -0.00023, -0.00023, -0.00022, -0.00022, -0.00022, -0.00021, -0.00021, -0.00021, -0.0002,
                    -0.0002, -0.0002, -0.00019, -0.00019, -0.00018, -0.00018, -0.00018, -0.00017, -0.00017, -0.00017,
                    -0.00016, -0.00016, -0.00016, -0.00016, -0.00015, -0.00015, -0.00015, -0.00014, -0.00014, -0.00014,
                    -0.00013, -0.00013, -0.00013, -0.00012, -0.00012, -0.00012, -0.00012, -0.00011, -0.00011, -0.00011,
                    -0.0001, -0.0001, -0.0001, -0.00009, -0.00009, -0.00009, -0.00009, -0.00008, -0.00008, -0.00008,
                    -0.00007, -0.00007, -0.00007, -0.00007, -0.00006, -0.00006, -0.00006, -0.00006, -0.00005, -0.00005,
                    -0.00005, -0.00004, -0.00004, -0.00004, -0.00004, -0.00003, -0.00003, -0.00003, -0.00003, -0.00002,
                    -0.00002, -0.00002, -0.00002, -0.00001, -0.00001, -0.000009, -0.000006, -0.000004, -0.000002,
                    0.000000916289, 0.000003, 0.000006, 0.000008, 0.00001, 0.00001, 0.00002, 0.00002, 0.00002, 0.00002,
                    0.00002, 0.00003, 0.00003, 0.00003, 0.00003, 0.00004, 0.00004, 0.00004, 0.00004, 0.00005, 0.00005,
                    0.00005, 0.00005, 0.00005, 0.00006, 0.00006, 0.00006, 0.00006, 0.00007, 0.00007, 0.00007, 0.00007,
                    0.00007, 0.00008, 0.00008, 0.00008, 0.00008, 0.00009, 0.00009, 0.00009, 0.00009, 0.00009, 0.0001,
                    0.0001, 0.0001, 0.0001, 0.0001, 0.00011, 0.00011, 0.00011, 0.00011, 0.00011, 0.00012, 0.00012,
                    0.00012, 0.00012, 0.00012, 0.00013, 0.00013, 0.00013, 0.00013, 0.00013, 0.00014, 0.00014, 0.00014,
                    0.00014, 0.00014, 0.00015, 0.00015, 0.00015, 0.00015, 0.00015, 0.00016, 0.00016, 0.00016, 0.00016,
                    0.00016, 0.00017, 0.00017, 0.00017, 0.00017, 0.00017, 0.00017, 0.00018, 0.00018, 0.00018, 0.00018,
                    0.00018, 0.00019, 0.00019, 0.00019, 0.00019, 0.00019, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002,
                    0.0002, 0.00021, 0.00021, 0.00021, 0.00021, 0.00021, 0.00022, 0.00022, 0.00022, 0.00022, 0.00022,
                    0.00022, 0.00023, 0.00023, 0.00023, 0.00023, 0.00023, 0.00023, 0.00024, 0.00024, 0.00024, 0.00024,
                    0.00024, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00026, 0.00026, 0.00026, 0.00026,
                    0.00026, 0.00026, 0.00027, 0.00027, 0.00027, 0.00027, 0.00027, 0.00028, 0.00028, 0.00028, 0.00028,
                    0.00028, 0.00028, 0.00029, 0.00029, 0.00029, 0.00029, 0.00029, 0.00029, 0.0003, 0.0003, 0.0003,
                    0.0003]
    fvwr_yu_array = [-0.00004, -0.19863, -0.3164, -0.39465, -0.45288, -0.49912, -0.53732, -0.56973, -0.59776, -0.62233,
                     -0.6441, -0.66355, -0.68105, -0.69686, -0.71123, -0.72431, -0.73629, -0.74732, -0.75754, -0.76705,
                     -0.77592, -0.78424, -0.79205, -0.7994, -0.80633, -0.81288, -0.81909, -0.82497, -0.83056, -0.83586,
                     -0.84091, -0.84572, -0.8503, -0.85468, -0.85885, -0.86284, -0.86665, -0.87029, -0.87378, -0.87712,
                     -0.88031, -0.88338, -0.88631, -0.88913, -0.89183, -0.89442, -0.89691, -0.8993, -0.9016, -0.9038,
                     -0.90593, -0.90797, -0.90993, -0.91182, -0.91365, -0.9154, -0.9171, -0.91873, -0.92032, -0.92184,
                     -0.92332, -0.92475, -0.92614, -0.92749, -0.9288, -0.93007, -0.93131, -0.93253, -0.93371, -0.93488,
                     -0.93602, -0.93714, -0.93825, -0.93935, -0.94042, -0.94148, -0.94253, -0.94356, -0.94457, -0.94556,
                     -0.94654, -0.9475, -0.94845, -0.94938, -0.95029, -0.95118, -0.95206, -0.95293, -0.95377, -0.9546,
                     -0.95541, -0.95621, -0.95699, -0.95776, -0.95851, -0.95924, -0.95996, -0.96066, -0.96135, -0.96202,
                     -0.96268, -0.96333, -0.96396, -0.96458, -0.96518, -0.96577, -0.96635, -0.96691, -0.96747, -0.96801,
                     -0.96854, -0.96906, -0.96956, -0.97006, -0.97055, -0.97102, -0.97149, -0.97195, -0.9724, -0.97284,
                     -0.97327, -0.97369, -0.97411, -0.97451, -0.97491, -0.97531, -0.97569, -0.97607, -0.97645, -0.97681,
                     -0.97718, -0.97753, -0.97788, -0.97823, -0.97857, -0.97891, -0.97924, -0.97957, -0.97989, -0.98021,
                     -0.98052, -0.98083, -0.98113, -0.98143, -0.98173, -0.98202, -0.9823, -0.98258, -0.98286, -0.98313,
                     -0.98339, -0.98366, -0.98391, -0.98417, -0.98441, -0.98466, -0.9849, -0.98514, -0.98537, -0.9856,
                     -0.98582, -0.98604, -0.98626, -0.98647, -0.98668, -0.98688, -0.98709, -0.98729, -0.98748, -0.98767,
                     -0.98786, -0.98805, -0.98823, -0.98841, -0.98858, -0.98876, -0.98893, -0.98909, -0.98926, -0.98942,
                     -0.98958, -0.98973, -0.98989, -0.99004, -0.99019, -0.99033, -0.99047, -0.99062, -0.99075, -0.99089,
                     -0.99102, -0.99116, -0.99129, -0.99141, -0.99154, -0.99166, -0.99178, -0.9919, -0.99202, -0.99214,
                     -0.99225, -0.99236, -0.99247, -0.99258, -0.99269, -0.99279, -0.9929, -0.993, -0.9931, -0.9932,
                     -0.99329, -0.99339, -0.99348, -0.99357, -0.99366, -0.99375, -0.99384, -0.99393, -0.99401, -0.99409,
                     -0.99418, -0.99426, -0.99434, -0.99442, -0.99449, -0.99457, -0.99464, -0.99472, -0.99479, -0.99486,
                     -0.99493, -0.995, -0.99507, -0.99513, -0.9952, -0.99526, -0.99533, -0.99539, -0.99545, -0.99551,
                     -0.99557, -0.99563, -0.99569, -0.99575, -0.9958, -0.99586, -0.99591, -0.99596, -0.99601, -0.99607,
                     -0.99612, -0.99617, -0.99622, -0.99626, -0.99631, -0.99636, -0.9964, -0.99645, -0.99649, -0.99654,
                     -0.99658, -0.99662, -0.99666, -0.99671, -0.99675, -0.99679, -0.99682, -0.99686, -0.9969, -0.99694,
                     -0.99698, -0.99701, -0.99705, -0.99708, -0.99712, -0.99715, -0.99719, -0.99722, -0.99725, -0.99729,
                     -0.99732, -0.99735, -0.99738, -0.99741, -0.99744, -0.99747, -0.9975, -0.99753, -0.99756, -0.99758,
                     -0.99761, -0.99764, -0.99766, -0.99769, -0.99772, -0.99774, -0.99777, -0.99779, -0.99782, -0.99784,
                     -0.99787, -0.99789, -0.99791, -0.99793, -0.99796, -0.99798, -0.998, -0.99802, -0.99804, -0.99806,
                     -0.99808, -0.9981, -0.99812, -0.99814, -0.99816, -0.99818, -0.9982, -0.99822, -0.99824, -0.99826,
                     -0.99827, -0.99829, -0.99831, -0.99833, -0.99834, -0.99836, -0.99837, -0.99839, -0.99841, -0.99842,
                     -0.99844, -0.99845, -0.99847, -0.99848, -0.9985, -0.99851, -0.99852, -0.99854, -0.99855, -0.99856,
                     -0.99858, -0.99859, -0.9986, -0.99861, -0.99863, -0.99864, -0.99865, -0.99866, -0.99867, -0.99869,
                     -0.9987, -0.99871, -0.99872, -0.99873, -0.99874, -0.99875, -0.99876, -0.99877, -0.99878, -0.99879,
                     -0.9988, -0.99881, -0.99882, -0.99883, -0.99884, -0.99884, -0.99885, -0.99886, -0.99887, -0.99888,
                     -0.99889, -0.99889, -0.9989, -0.99891, -0.99892, -0.99892, -0.99893, -0.99894, -0.99894, -0.99895,
                     -0.99896, -0.99896, -0.99897, -0.99898, -0.99898, -0.99899, -0.999, -0.999, -0.99901, -0.99901,
                     -0.99902, -0.99902, -0.99903, -0.99903, -0.99904, -0.99904, -0.99905, -0.99905, -0.99906, -0.99906,
                     -0.99907, -0.99907, -0.99908, -0.99908, -0.99909, -0.99909, -0.99909, -0.9991, -0.9991, -0.99911,
                     -0.99911, -0.99911, -0.99912, -0.99912, -0.99912, -0.99913, -0.99913, -0.99913, -0.99914, -0.99914,
                     -0.99914, -0.99915, -0.99915, -0.99915, -0.99915, -0.99916, -0.99916, -0.99916, -0.99917, -0.99917,
                     -0.99917, -0.99917, -0.99917, -0.99918, -0.99918, -0.99918, -0.99918, -0.99919, -0.99919, -0.99919,
                     -0.99919, -0.99919, -0.9992, -0.9992, -0.9992, -0.9992, -0.9992, -0.9992, -0.99921, -0.99921,
                     -0.99921, -0.99921, -0.99921, -0.99921, -0.99922, -0.99922, -0.99922, -0.99922, -0.99922, -0.99922,
                     -0.99922, -0.99923, -0.99923, -0.99923, -0.99923, -0.99923, -0.99923, -0.99923, -0.99923, -0.99924,
                     -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99925, -0.99925, -0.99925,
                     -0.99925, -0.99925, -0.95533, -0.83911, -0.73646, -0.64966, -0.57711, -0.51623, -0.46469, -0.42066,
                     -0.38273, -0.34983, -0.32112, -0.29597, -0.27387, -0.25441, -0.23715, -0.22172, -0.20782, -0.19525,
                     -0.18381, -0.17338, -0.16383, -0.15508, -0.14703, -0.13961, -0.13278, -0.12646, -0.12063, -0.11523,
                     -0.11023, -0.10559, -0.10129, -0.0973, -0.0936, -0.09015, -0.08695, -0.08396, -0.08118, -0.07858,
                     -0.07615, -0.07387, -0.07173, -0.06971, -0.06781, -0.06599, -0.06427, -0.06261, -0.06101, -0.05945,
                     -0.05793, -0.05644, -0.05499, -0.05356, -0.05218, -0.05082, -0.04951, -0.04823, -0.04698, -0.04577,
                     -0.04459, -0.04345, -0.04234, -0.04127, -0.04023, -0.03922, -0.03824, -0.03729, -0.03638, -0.03549,
                     -0.03463, -0.0338, -0.03299, -0.03221, -0.03146, -0.03073, -0.03002, -0.02933, -0.02866, -0.02802,
                     -0.02739, -0.02678, -0.02619, -0.02561, -0.02506, -0.02451, -0.02398, -0.02347, -0.02297, -0.02248,
                     -0.02201, -0.02154, -0.02109, -0.02065, -0.02022, -0.0198, -0.01939, -0.01899, -0.01859, -0.01821,
                     -0.01783, -0.01746, -0.0171, -0.01675, -0.01641, -0.01607, -0.01574, -0.01542, -0.01511, -0.0148,
                     -0.01451, -0.01421, -0.01393, -0.01365, -0.01338, -0.01311, -0.01285, -0.01259, -0.01234, -0.0121,
                     -0.01186, -0.01163, -0.0114, -0.01118, -0.01096, -0.01074, -0.01054, -0.01033, -0.01013, -0.00993,
                     -0.00974, -0.00955, -0.00937, -0.00919, -0.00901, -0.00884, -0.00867, -0.0085, -0.00834, -0.00817,
                     -0.00802, -0.00786, -0.00771, -0.00756, -0.00742, -0.00728, -0.00714, -0.007, -0.00686, -0.00673,
                     -0.0066, -0.00647, -0.00635, -0.00622, -0.0061, -0.00599, -0.00587, -0.00575, -0.00564, -0.00553,
                     -0.00542, -0.00532, -0.00521, -0.00511, -0.00501, -0.00491, -0.00481, -0.00472, -0.00462, -0.00453,
                     -0.00444, -0.00435, -0.00426, -0.00417, -0.00409, -0.00401, -0.00392, -0.00384, -0.00376, -0.00369,
                     -0.00361, -0.00353, -0.00346, -0.00339, -0.00331, -0.00324, -0.00317, -0.00311, -0.00304, -0.00297,
                     -0.00291, -0.00284, -0.00278, -0.00272, -0.00266, -0.0026, -0.00254, -0.00248, -0.00243, -0.00237,
                     -0.00232, -0.00226, -0.00221, -0.00216, -0.0021, -0.00205, -0.002, -0.00196, -0.00191, -0.00186,
                     -0.00181, -0.00177, -0.00172, -0.00168, -0.00164, -0.00159, -0.00155, -0.00151, -0.00147, -0.00143,
                     -0.00139, -0.00135, -0.00131, -0.00128, -0.00124, -0.00121, -0.00117, -0.00113, -0.0011, -0.00107,
                     -0.00103, -0.001, -0.00097, -0.00094, -0.00091, -0.00088, -0.00085, -0.00082, -0.00079, -0.00076,
                     -0.00073, -0.0007, -0.00068, -0.00065, -0.00062, -0.0006, -0.00057, -0.00055, -0.00052, -0.0005,
                     -0.00047, -0.00045, -0.00043, -0.0004, -0.00038, -0.00036, -0.00034, -0.00032, -0.00029, -0.00027,
                     -0.00025, -0.00023, -0.00021, -0.0002, -0.00018, -0.00016, -0.00014, -0.00012, -0.0001, -0.00009,
                     -0.00007, -0.00005, -0.00004, -0.00002, -0.000003, 0.00001, 0.00003, 0.00004, 0.00006, 0.00007,
                     0.00009, 0.0001, 0.00012, 0.00013, 0.00014, 0.00016, 0.00017, 0.00018, 0.00019, 0.00021, 0.00022,
                     0.00023, 0.00024, 0.00025, 0.00026, 0.00027, 0.00029, 0.0003, 0.00031, 0.00032, 0.00033, 0.00034,
                     0.00035, 0.00035, 0.00036, 0.00037, 0.00038, 0.00039, 0.0004, 0.00041, 0.00041, 0.00042, 0.00043,
                     0.00044, 0.00044, 0.00045, 0.00046, 0.00046, 0.00047, 0.00048, 0.00048, 0.00049, 0.0005, 0.0005,
                     0.00051, 0.00051, 0.00052, 0.00052, 0.00053, 0.00053, 0.00054, 0.00054, 0.00055, 0.00055, 0.00056,
                     0.00056, 0.00056, 0.00057, 0.00057, 0.00058, 0.00058, 0.00058, 0.00059, 0.00059, 0.00059, 0.00059,
                     0.0006, 0.0006, 0.0006, 0.0006, 0.00061, 0.00061, 0.00061, 0.00061, 0.00061, 0.00062, 0.00062,
                     0.00062, 0.00062, 0.00062, 0.00062, 0.00062, 0.00063, 0.00063, 0.00063, 0.00063, 0.00063, 0.00063,
                     0.00063, 0.00063, 0.00063, 0.00063, 0.00063, 0.00063, 0.00063, 0.00063, 0.00063, 0.00063, 0.00063,
                     0.00063, 0.00063, 0.00063, 0.00063, 0.00063, 0.00062, 0.00062, 0.00062, 0.00062, 0.00062, 0.00062,
                     0.00062, 0.00062, 0.00061, 0.00061, 0.00061, 0.00061, 0.00061, 0.00061, 0.0006, 0.0006, 0.0006,
                     0.0006, 0.0006, 0.00059, 0.00059, 0.00059, 0.00059, 0.00058, 0.00058, 0.00058, 0.00058, 0.00057,
                     0.00057, 0.00057, 0.00057, 0.00056, 0.00056, 0.00056, 0.00055, 0.00055, 0.00055, 0.00055, 0.00054,
                     0.00054, 0.00054, 0.00053, 0.00053, 0.00053, 0.00052, 0.00052, 0.00052, 0.00051, 0.00051, 0.00051,
                     0.0005, 0.0005, 0.00049, 0.00049, 0.00049, 0.00048, 0.00048, 0.00048, 0.00047, 0.00047, 0.00046,
                     0.00046, 0.00046, 0.00045, 0.00045, 0.00044, 0.00044, 0.00044, 0.00043, 0.00043, 0.00042, 0.00042,
                     0.00042, 0.00041, 0.00041, 0.0004, 0.0004, 0.0004, 0.00039, 0.00039, 0.00038, 0.00038, 0.00037,
                     0.00037, 0.00037, 0.00036, 0.00036, 0.00035, 0.00035, 0.00034, 0.00034, 0.00033, 0.00033, 0.00033,
                     0.00032, 0.00032, 0.00031, 0.00031, 0.0003]
    rans_yu_array = [0.00591, -0.08345, -0.16921, -0.24852, -0.32002, -0.38354, -0.43953, -0.48877, -0.53205, -0.57014,
                     -0.6037, -0.6333, -0.65944, -0.68251, -0.70293, -0.72107, -0.73725, -0.75175, -0.76482, -0.77664,
                     -0.78739, -0.7972, -0.80619, -0.81446, -0.82209, -0.82916, -0.83573, -0.84184, -0.84755, -0.85289,
                     -0.85791, -0.86262, -0.86706, -0.87125, -0.87521, -0.87896, -0.88252, -0.88591, -0.88912, -0.89219,
                     -0.89512, -0.89791, -0.90059, -0.90315, -0.90561, -0.90796, -0.91023, -0.91241, -0.9145, -0.91653,
                     -0.91848, -0.92037, -0.92219, -0.92396, -0.92567, -0.92733, -0.92894, -0.93051, -0.93202, -0.9335,
                     -0.93493, -0.93633, -0.93769, -0.93901, -0.9403, -0.94156, -0.94278, -0.94397, -0.94514, -0.94627,
                     -0.94738, -0.94846, -0.94952, -0.95055, -0.95156, -0.95255, -0.95351, -0.95445, -0.95537, -0.95627,
                     -0.95715, -0.95802, -0.95886, -0.95969, -0.96049, -0.96128, -0.96206, -0.96282, -0.96356, -0.96429,
                     -0.965, -0.9657, -0.96638, -0.96705, -0.96771, -0.96836, -0.96899, -0.96961, -0.97022, -0.97081,
                     -0.9714, -0.97197, -0.97253, -0.97308, -0.97362, -0.97416, -0.97468, -0.97519, -0.97569, -0.97619,
                     -0.97667, -0.97715, -0.97762, -0.97808, -0.97853, -0.97897, -0.97941, -0.97984, -0.98026, -0.98068,
                     -0.98109, -0.98149, -0.98189, -0.98228, -0.98267, -0.98305, -0.98342, -0.98379, -0.98416, -0.98452,
                     -0.98487, -0.98522, -0.98557, -0.98591, -0.98625, -0.98657, -0.9869, -0.98722, -0.98753, -0.98783,
                     -0.98813, -0.98843, -0.98871, -0.98899, -0.98927, -0.98954, -0.9898, -0.99005, -0.9903, -0.99054,
                     -0.99078, -0.99101, -0.99123, -0.99145, -0.99166, -0.99186, -0.99206, -0.99225, -0.99244, -0.99262,
                     -0.9928, -0.99297, -0.99314, -0.9933, -0.99345, -0.9936, -0.99375, -0.99389, -0.99403, -0.99416,
                     -0.99429, -0.99441, -0.99453, -0.99465, -0.99476, -0.99487, -0.99498, -0.99508, -0.99518, -0.99527,
                     -0.99537, -0.99546, -0.99555, -0.99563, -0.99571, -0.99579, -0.99587, -0.99594, -0.99602, -0.99609,
                     -0.99616, -0.99622, -0.99629, -0.99635, -0.99641, -0.99647, -0.99653, -0.99658, -0.99663, -0.99669,
                     -0.99674, -0.99679, -0.99684, -0.99688, -0.99693, -0.99697, -0.99702, -0.99706, -0.9971, -0.99714,
                     -0.99718, -0.99722, -0.99725, -0.99729, -0.99732, -0.99736, -0.99739, -0.99742, -0.99746, -0.99749,
                     -0.99752, -0.99755, -0.99758, -0.99761, -0.99763, -0.99766, -0.99769, -0.99771, -0.99774, -0.99776,
                     -0.99779, -0.99781, -0.99784, -0.99786, -0.99788, -0.99791, -0.99793, -0.99795, -0.99797, -0.99799,
                     -0.99801, -0.99803, -0.99805, -0.99807, -0.99809, -0.99811, -0.99813, -0.99814, -0.99816, -0.99818,
                     -0.9982, -0.99821, -0.99823, -0.99825, -0.99826, -0.99828, -0.99829, -0.99831, -0.99832, -0.99834,
                     -0.99835, -0.99837, -0.99838, -0.9984, -0.99841, -0.99842, -0.99844, -0.99845, -0.99846, -0.99848,
                     -0.99849, -0.9985, -0.99851, -0.99852, -0.99854, -0.99855, -0.99856, -0.99857, -0.99858, -0.99859,
                     -0.9986, -0.99861, -0.99862, -0.99863, -0.99865, -0.99866, -0.99866, -0.99867, -0.99868, -0.99869,
                     -0.9987, -0.99871, -0.99872, -0.99873, -0.99874, -0.99875, -0.99876, -0.99876, -0.99877, -0.99878,
                     -0.99879, -0.9988, -0.9988, -0.99881, -0.99882, -0.99883, -0.99883, -0.99884, -0.99885, -0.99886,
                     -0.99886, -0.99887, -0.99888, -0.99888, -0.99889, -0.99889, -0.9989, -0.99891, -0.99891, -0.99892,
                     -0.99893, -0.99893, -0.99894, -0.99894, -0.99895, -0.99895, -0.99896, -0.99896, -0.99897, -0.99898,
                     -0.99898, -0.99899, -0.99899, -0.99899, -0.999, -0.999, -0.99901, -0.99901, -0.99902, -0.99902,
                     -0.99903, -0.99903, -0.99904, -0.99904, -0.99904, -0.99905, -0.99905, -0.99906, -0.99906, -0.99906,
                     -0.99907, -0.99907, -0.99907, -0.99908, -0.99908, -0.99908, -0.99909, -0.99909, -0.99909, -0.9991,
                     -0.9991, -0.9991, -0.99911, -0.99911, -0.99911, -0.99912, -0.99912, -0.99912, -0.99912, -0.99913,
                     -0.99913, -0.99913, -0.99913, -0.99914, -0.99914, -0.99914, -0.99914, -0.99915, -0.99915, -0.99915,
                     -0.99915, -0.99916, -0.99916, -0.99916, -0.99916, -0.99916, -0.99917, -0.99917, -0.99917, -0.99917,
                     -0.99917, -0.99918, -0.99918, -0.99918, -0.99918, -0.99918, -0.99918, -0.99919, -0.99919, -0.99919,
                     -0.99919, -0.99919, -0.99919, -0.99919, -0.9992, -0.9992, -0.9992, -0.9992, -0.9992, -0.9992,
                     -0.9992, -0.9992, -0.99921, -0.99921, -0.99921, -0.99921, -0.99921, -0.99921, -0.99921, -0.99921,
                     -0.99921, -0.99922, -0.99922, -0.99922, -0.99922, -0.99922, -0.99922, -0.99922, -0.99922, -0.99922,
                     -0.99922, -0.99922, -0.99922, -0.99923, -0.99923, -0.99923, -0.99923, -0.99923, -0.99923, -0.99923,
                     -0.99923, -0.99923, -0.99923, -0.99923, -0.99923, -0.99923, -0.99923, -0.99923, -0.99923, -0.99924,
                     -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924,
                     -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924, -0.99924,
                     -0.99924, -0.99924, -0.99925, -0.99925, -0.99925, -0.99925, -0.99925, -0.99925, -0.99925, -0.99925,
                     -0.99925, -0.99925, -0.96205, -0.84423, -0.74245, -0.65835, -0.58914, -0.53154, -0.48293, -0.44137,
                     -0.40544, -0.3741, -0.34657, -0.32224, -0.30063, -0.2813, -0.26388, -0.24808, -0.23368, -0.22049,
                     -0.20837, -0.19719, -0.18684, -0.17725, -0.16832, -0.16001, -0.15224, -0.14498, -0.13818, -0.1318,
                     -0.1258, -0.12016, -0.11485, -0.10984, -0.10511, -0.10065, -0.09642, -0.09242, -0.08864, -0.08505,
                     -0.08164, -0.0784, -0.07533, -0.07241, -0.06962, -0.06698, -0.06445, -0.06204, -0.05975, -0.05755,
                     -0.05546, -0.05345, -0.05154, -0.0497, -0.04794, -0.04625, -0.04464, -0.04309, -0.0416, -0.04017,
                     -0.03879, -0.03748, -0.03621, -0.03499, -0.03382, -0.0327, -0.03162, -0.03058, -0.02959, -0.02863,
                     -0.02771, -0.02682, -0.02597, -0.02515, -0.02437, -0.02361, -0.02289, -0.02219, -0.02152, -0.02088,
                     -0.02026, -0.01967, -0.0191, -0.01856, -0.01803, -0.01753, -0.01705, -0.01659, -0.01615, -0.01573,
                     -0.01532, -0.01494, -0.01457, -0.01421, -0.01387, -0.01355, -0.01324, -0.01294, -0.01266, -0.01239,
                     -0.01213, -0.01188, -0.01164, -0.01142, -0.0112, -0.011, -0.0108, -0.01061, -0.01043, -0.01026,
                     -0.01009, -0.00993, -0.00978, -0.00963, -0.00949, -0.00935, -0.00922, -0.00909, -0.00896, -0.00884,
                     -0.00871, -0.00859, -0.00847, -0.00836, -0.00824, -0.00812, -0.008, -0.00789, -0.00777, -0.00766,
                     -0.00755, -0.00743, -0.00732, -0.00721, -0.00711, -0.007, -0.00689, -0.00679, -0.00669, -0.00658,
                     -0.00649, -0.00639, -0.00629, -0.0062, -0.0061, -0.00601, -0.00592, -0.00583, -0.00574, -0.00566,
                     -0.00558, -0.00549, -0.00541, -0.00533, -0.00526, -0.00518, -0.00511, -0.00503, -0.00496, -0.00489,
                     -0.00482, -0.00475, -0.00469, -0.00462, -0.00456, -0.0045, -0.00444, -0.00437, -0.00432, -0.00426,
                     -0.0042, -0.00415, -0.00409, -0.00404, -0.00399, -0.00393, -0.00388, -0.00383, -0.00378, -0.00374,
                     -0.00369, -0.00364, -0.0036, -0.00355, -0.00351, -0.00347, -0.00343, -0.00338, -0.00334, -0.0033,
                     -0.00326, -0.00323, -0.00319, -0.00315, -0.00311, -0.00308, -0.00304, -0.00301, -0.00297, -0.00294,
                     -0.00291, -0.00287, -0.00284, -0.00281, -0.00278, -0.00275, -0.00272, -0.00269, -0.00266, -0.00263,
                     -0.0026, -0.00257, -0.00255, -0.00252, -0.00249, -0.00247, -0.00244, -0.00242, -0.00239, -0.00237,
                     -0.00234, -0.00232, -0.00229, -0.00227, -0.00225, -0.00222, -0.0022, -0.00218, -0.00216, -0.00214,
                     -0.00212, -0.0021, -0.00208, -0.00205, -0.00203, -0.00202, -0.002, -0.00198, -0.00196, -0.00194,
                     -0.00192, -0.0019, -0.00188, -0.00187, -0.00185, -0.00183, -0.00181, -0.0018, -0.00178, -0.00176,
                     -0.00175, -0.00173, -0.00172, -0.0017, -0.00168, -0.00167, -0.00165, -0.00164, -0.00162, -0.00161,
                     -0.00159, -0.00158, -0.00156, -0.00155, -0.00154, -0.00152, -0.00151, -0.00149, -0.00148, -0.00147,
                     -0.00145, -0.00144, -0.00143, -0.00141, -0.0014, -0.00139, -0.00138, -0.00136, -0.00135, -0.00134,
                     -0.00133, -0.00132, -0.0013, -0.00129, -0.00128, -0.00127, -0.00126, -0.00125, -0.00123, -0.00122,
                     -0.00121, -0.0012, -0.00119, -0.00118, -0.00117, -0.00116, -0.00115, -0.00114, -0.00113, -0.00112,
                     -0.00111, -0.0011, -0.00109, -0.00108, -0.00107, -0.00106, -0.00105, -0.00104, -0.00103, -0.00102,
                     -0.00101, -0.001, -0.00099, -0.00098, -0.00097, -0.00096, -0.00095, -0.00094, -0.00093, -0.00093,
                     -0.00092, -0.00091, -0.0009, -0.00089, -0.00088, -0.00087, -0.00087, -0.00086, -0.00085, -0.00084,
                     -0.00083, -0.00082, -0.00082, -0.00081, -0.0008, -0.00079, -0.00078, -0.00078, -0.00077, -0.00076,
                     -0.00075, -0.00075, -0.00074, -0.00073, -0.00072, -0.00072, -0.00071, -0.0007, -0.00069, -0.00069,
                     -0.00068, -0.00067, -0.00066, -0.00066, -0.00065, -0.00064, -0.00064, -0.00063, -0.00062, -0.00062,
                     -0.00061, -0.0006, -0.0006, -0.00059, -0.00058, -0.00058, -0.00057, -0.00056, -0.00056, -0.00055,
                     -0.00054, -0.00054, -0.00053, -0.00052, -0.00052, -0.00051, -0.0005, -0.0005, -0.00049, -0.00049,
                     -0.00048, -0.00047, -0.00047, -0.00046, -0.00046, -0.00045, -0.00044, -0.00044, -0.00043, -0.00043,
                     -0.00042, -0.00041, -0.00041, -0.0004, -0.0004, -0.00039, -0.00038, -0.00038, -0.00037, -0.00037,
                     -0.00036, -0.00036, -0.00035, -0.00035, -0.00034, -0.00033, -0.00033, -0.00032, -0.00032, -0.00031,
                     -0.00031, -0.0003, -0.0003, -0.00029, -0.00029, -0.00028, -0.00027, -0.00027, -0.00026, -0.00026,
                     -0.00025, -0.00025, -0.00024, -0.00024, -0.00023, -0.00023, -0.00022, -0.00022, -0.00021, -0.00021,
                     -0.0002, -0.0002, -0.00019, -0.00019, -0.00018, -0.00018, -0.00017, -0.00017, -0.00016, -0.00016,
                     -0.00015, -0.00015, -0.00014, -0.00014, -0.00013, -0.00013, -0.00012, -0.00012, -0.00011, -0.00011,
                     -0.0001, -0.0001, -0.00009, -0.00009, -0.00008, -0.00008, -0.00007, -0.00007, -0.00006, -0.00006,
                     -0.00005, -0.00005, -0.00004, -0.00004, -0.00003, -0.00003, -0.00002, -0.00002, -0.00001,
                     -0.000009, -0.000005, 0.000000190806, 0.000005, 0.00001, 0.00001, 0.00002, 0.00002, 0.00003,
                     0.00003, 0.00004, 0.00004, 0.00005, 0.00005]

    # subset_tau_array = []
    # subset_deltaV_array = []
    # for tdx in range(tau_array.shape[0]):
    #     if tau_array[tdx] >= tau_switch:
    #         subset_tau_array += [tau_array[tdx]]
    #         subset_deltaV_array += [deltaV_array[tdx]]
    #
    # base_name = "actuator_uaxi_verification"
    #
    # import os
    # import matplotlib
    # if "MPLBACKEND" not in os.environ:
    #     matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt
    # fig, ax1 = plt.subplots()
    # color1 = 'tab:red'
    # ax1.set_ylabel('thrust coefficient [-]', color=color1)
    # ax1.plot(tau_array, CT_array, label='CT', color=color1)
    # ax1.tick_params(axis='y', labelcolor=color1)
    #
    # ax2 = ax1.twinx()
    # color2 = 'tab:blue'
    # ax2.set_ylabel('axial induction factor [-]', color=color2)
    # ax2.plot(tau_array, a_array, label='a', color=color2)
    # ax2.tick_params(axis='y', labelcolor=color2)
    # plt.savefig(base_name + "_ct_and_a.pdf", format="pdf")
    #
    # fig = plt.figure()
    # plt.plot(x_yu_array, qaxi_yu_array, label='qaxi_yu', color='blue')
    # plt.plot(x_yu_array, uaxi_yu_array, label='uaxi_yu', color='green')
    # plt.plot(x_yu_array, oye_yu_array, label='oye_yu', color='red')
    # plt.plot(x_yu_array, ecn_yu_array, label='ecn_yu', color='cyan')
    # plt.plot(x_yu_array, vtn_yu_array, label='vtn_yu', color='magenta')
    # plt.plot(x_yu_array, fvwr_yu_array, label='fwrr_yu', color='yellow')
    # plt.plot(x_yu_array, rans_yu_array, label='rans_yu', color='gray')
    # # notice that deltaV in figure 6 seems to have the sign switched.
    # plt.plot(subset_tau_array, subset_deltaV_array, label='uaxi_awebox', color='black')
    # plt.legend()
    # plt.savefig(base_name + "_deltav.pdf", format="pdf")
    #
    # plt.show()

    return None

def get_unsteady_asym_pitt_peters_residual(model_options, atmos, wind, system_variables, parameters, outputs, parent, architecture, label, scaling):
    variables_si = system_variables['SI']

    a_all = actuator_flow.get_a_all_var(variables_si, parent, label)
    da_dt_all = actuator_flow.get_da_all_var(variables_si, parent, label)
    a_all_ref = actuator_flow.get_a_all_ref(model_options, parent, scaling)
    da_dt_all_ref = actuator_flow.get_da_all_ref(model_options, parent, scaling)

    c_all, c_numer, c_denom, c_all_ref, c_num_ref, c_den_ref = actuator_coeff.get_c_all_components(model_options, atmos, wind, system_variables, parameters, outputs, parent, architecture, scaling)

    LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_val(model_options, atmos, wind, system_variables,
                                                                         outputs, parameters, parent, architecture,
                                                                         scaling, label)

    MM = actuator_coeff.get_MM_matrix()
    t_star, t_star_num, t_star_den = actuator_coeff.get_t_star(variables_si, parameters, parent)

    resi_unscaled, _, _, _ = assemble_pitt_peters_style_residual(a_all, da_dt_all, t_star, t_star_num, t_star_den, MM, LLinv, LLinv_num, LLinv_denom, c_all, c_numer, c_denom)

    LLinv_ref, LLinv_num_ref, LLinv_denom_ref = actuator_coeff.get_LL_inv_matrix_ref(model_options, parent, scaling)
    t_star_ref, t_star_num_ref, t_star_den_ref = actuator_coeff.get_t_star_ref(model_options, parameters, parent, scaling)

    _, term_1_ref, term_2_ref, term_3_ref = assemble_pitt_peters_style_residual(a_all_ref, da_dt_all_ref, t_star_ref, t_star_num_ref, t_star_den_ref, MM, LLinv_ref, LLinv_num_ref, LLinv_denom_ref, c_all_ref, c_num_ref, c_den_ref)

    # resi_reference = term_1_ref
    # term_ref_name = 'term_1'

    resi_reference = term_2_ref
    term_ref_name = 'term_2'

    # resi_reference = term_3_ref
    # term_ref_name = 'term_3'

    resi = scale_resi(resi_unscaled, resi_reference, term_ref_name)
    return resi


def test_signs_of_qasym_with_nonzero_moment_from_Sun(eps=1e-4):

    # H. Sun, A. Sciacchitano, and W. Yu,
    # “ An Engineering Model for Static Yawed Wind Turbines Based on Actuator Line Simulations and Symbolic Regression,” Wind Energy 29, no. 5 (2026): e70118,
    # https://doi.org/10.1002/we.70118.

    u_zero = 8 # m/s, page 5
    diameter = 126. #m, pg. 3
    gamma_yaw = 25. * np.pi / 180. # fig. 18b
    hub_radius = 1.5/2. #m, pg. 5
    rho = 1.225 # kg/m^3, pg. 3
    pitch = 0 # page 7

    area = np.pi * (diameter / 2.)**2.
    exterior_radius = diameter/2.
    actuator_dynamic_pressure = 0.5 * rho * u_zero**2.
    radius_bar = (hub_radius + exterior_radius) / 2.
    wingspan = (exterior_radius - radius_bar) * 2.

    ct = 0.867977 # digitized from figure 2
    thrust = ct * actuator_dynamic_pressure * area * cas.cos(gamma_yaw)**2. # see (10)

    # interpreted from fig. 3
    n_hat = cas.cos(gamma_yaw) * vect_op.xhat_dm() + cas.sin(gamma_yaw) * vect_op.yhat_dm()

    u_hat = vect_op.xhat_dm()
    dcm_act, dcm_wind = actuator_flow.construct_actuator_and_wind_reference_frames_from_n_and_u(n_hat, u_hat)
    y_rotor_hat = dcm_act[:, 1]
    z_rotor_hat = dcm_act[:, 2]

    moment_y = -436224.
    moment_z = -307100.

    approx_a0 = 0.

    c_all, c_numer, c_denom = actuator_coeff.assemble_c_components_from_pieces(thrust, moment_y, moment_z, actuator_dynamic_pressure, area, radius_bar, wingspan)

    veri_test_table = {
        'name': 'Sun',
        'YawAngle': gamma_yaw * 180./np.pi,
        'nhat': n_hat,
        'uhat': u_hat,
        'RotorMisalignmentAngle': gamma_yaw * 180./np.pi,
        'CT': c_all[0],
        'Cmy': c_all[1],
        'Cmz': c_all[2],
        'ExteriorRadius': exterior_radius,
        'AirDensity': rho,
        'RotorApparentSpeed': u_zero
    }
    print_op.print_dict_as_table(veri_test_table)


    step_history = {0: {'a0': approx_a0, 'acos': 0, 'asin': 0}}
    for step in range(20):
        corr = actuator_flow.get_corr_val_glauert(approx_a0, gamma_yaw)
        chi = actuator_flow.get_wake_angle_chi_jimenez(gamma_yaw, thrust, area, actuator_dynamic_pressure)
        # chi = actuator_flow.get_wake_angle_chi_coleman(gamma_yaw, approx_a0)
        LL_matr = actuator_coeff.get_LL_matrix_from_corr_and_chi(corr, chi)

        a_all = cas.mtimes(LL_matr, c_all)
        a_dict = actuator_flow.unstack_a_all(a_all)
        step_history[step] = a_dict

        approx_a0 = a_dict['a0']

    plot_induction_contour(a_dict, u_zero, exterior_radius, y_rotor_hat, z_rotor_hat, 'Sun') #, cmap='turbo_r')
    # print_op.print_dict_as_table(step_history)
    # for positive gamma:
    # [ a0   ]      [ + & 0 & - ] [ + ]   [ + ]
    # [ acos ] =    [ 0 & - & 0 ] [ + ] = [ - ]
    # [ asin ]      [ + & 0 & - ] [ - ] = [ + ]
    white_line_in_same_quadrant_as_paper_plot = (a_dict['a0'] > 0) and (a_dict['acos'] < 0) and (a_dict['asin'] > 0)
    if not white_line_in_same_quadrant_as_paper_plot:
        message = 'tilt and yaw moment not turning the induction contour in expected direction'
        print_op.log_and_raise_error(message)


    return None





def test_signs_of_qasym_with_nonzero_moment_from_Ma(eps=1e-4):

    # Ma L, Li Y, Zhou L, Yang D, Shen X and Du Z (2025)
    # Study on the aeroelastic performance of 15 MW wind turbine under yaw condition. Front. Energy Res. 13:1571567. doi: 10.3389/fenrg.2025.1571567
    # https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2025.1571567/full
    #

    u_zero = 10.59 # m/s, page 03
    diameter = 240. #m, page 03
    gamma_yaw = 20. * np.pi / 180. #pg 05
    hub_radius = 0.2 * (diameter/2.) # fig. 8, pg. 8
    omega = 7.55 * 2. * np.pi /60. # rad/s, table 1

    area = np.pi * (diameter / 2.)**2.
    exterior_radius = diameter/2.
    rho = 1.225 #guessed
    actuator_dynamic_pressure = 0.5 * rho * u_zero**2.
    radius_bar = (hub_radius + exterior_radius) / 2.
    wingspan = (exterior_radius - radius_bar) * 2.

    # value averaged from digitized (rigid + yaw) curves of Figure 5
    thrust = 2.28859147311828e6 #N
    tilt_moment = 2.23016428571429e6 #Nm
    yaw_moment = -3.05970833333333e6 #Nm

    # tilt_angle = 6. * np.pi / 180. # https://docs.nlr.gov/docs/fy20osti/75698.pdf
    tilt_angle = 0. # because it's not mentioned in the paper

    # see coordinates in Figure 18. earth fixed x, y, and z are defined the same.
    # but normal force is greatest between 180 (down) and 270 (upwind side) degrees?
    # ignore the x-y labels in figure 3, they don't match the coordinates from Fig. 9

    n_vec = vect_op.xhat_dm() + cas.tan(gamma_yaw) * vect_op.yhat_dm() + cas.tan(tilt_angle) * vect_op.zhat_dm()
    n_hat = vect_op.normalize(n_vec)
    u_hat = vect_op.xhat_dm()
    dcm_act, dcm_wind = actuator_flow.construct_actuator_and_wind_reference_frames_from_n_and_u(n_hat, u_hat)
    y_rotor_hat = dcm_act[:, 1]
    z_rotor_hat = dcm_act[:, 2]

    moment_y = -tilt_moment
    moment_z = yaw_moment

    # ct = thrust / area / actuator_dynamic_pressure
    # cmy = moment_y / area / actuator_dynamic_pressure / exterior_radius
    # cmz = moment_z / area / actuator_dynamic_pressure / exterior_radius
    # print('ct: ', ct)
    # print('cmy: ', cmy)
    # print('cmz: ', cmz)

    approx_avg_normal_ind_velocity = (-2.5 + (-6.)) / 2.
    approx_a0 = -1. * approx_avg_normal_ind_velocity / u_zero
    # approx_a0 = 0.5 - (1 - (thrust / actuator_dynamic_pressure / area))**0.5 / 2.
    # approx_a0 = cas.cos(gamma_yaw) / 2. - 0.5 * (-(thrust / actuator_dynamic_pressure / area) + cas.cos(gamma_yaw)**2.)**0.5

    c_all, c_numer, c_denom = actuator_coeff.assemble_c_components_from_pieces(thrust, moment_y, moment_z, actuator_dynamic_pressure, area, radius_bar, wingspan)

    veri_test_table = {
        'name': 'Ma',
        'YawAngle': gamma_yaw * 180./np.pi,
        'nhat': n_hat,
        'uhat': u_hat,
        'RotorMisalignmentAngle': gamma_yaw * 180./np.pi,
        'CT': c_all[0],
        'Cmy': c_all[1],
        'Cmz': c_all[2],
        'ExteriorRadius': exterior_radius,
        'AirDensity': rho,
        'RotorApparentSpeed': u_zero
    }
    print_op.print_dict_as_table(veri_test_table)


    step_history = {0: {'a0': approx_a0, 'acos': 0, 'asin': 0}}
    for step in range(20):
        corr = actuator_flow.get_corr_val_glauert(approx_a0, gamma_yaw)
        chi = actuator_flow.get_wake_angle_chi_jimenez(gamma_yaw, thrust, area, actuator_dynamic_pressure)
        # chi = actuator_flow.get_wake_angle_chi_coleman(gamma_yaw, approx_a0)
        LL_matr = actuator_coeff.get_LL_matrix_from_corr_and_chi(corr, chi)

        a_all = cas.mtimes(LL_matr, c_all)
        a_dict = actuator_flow.unstack_a_all(a_all)
        step_history[step] = a_dict

        approx_a0 = a_dict['a0']

    plot_induction_contour(a_dict, u_zero, exterior_radius, y_rotor_hat, z_rotor_hat, 'Ma')

    # print_op.print_dict_as_table(step_history)
    # for positive gamma:
    # [ a0   ]      [ + & 0 & - ] [ + ]   [ + ]
    # [ acos ] =    [ 0 & - & 0 ] [ + ] = [ - ]
    # [ asin ]      [ + & 0 & - ] [ - ] = [ + ]
    white_line_in_same_quadrant_as_paper_plot = (a_dict['a0'] > 0) and (a_dict['acos'] < 0) and (a_dict['asin'] > 0)
    if not white_line_in_same_quadrant_as_paper_plot:
        message = 'tilt and yaw moment not turning the induction contour in expected direction'
        print_op.log_and_raise_error(message)


    return None



def test_helicopter_values_for_qasym(thresh=2.0):

    # https://link.springer.com/article/10.1007/s13272-023-00680-z
    # Kostek 2024

    exterior_radius = 0.318 / 2. # (*page 645 *)
    rho = 1.225 #(*kg / m ^ 3 *)
    omega = 5400. * 2. *np.pi / 60. # (*rad / s *)
    u_zero = 12.9 #(*m / s *)
    tilt = -10. * np.pi / 180.

    hub_radius = 0. #40.e-3 #fig 5

    area = np.pi * exterior_radius**2.
    actuator_dynamic_pressure = 0.5 * rho * u_zero**2.
    radius_bar = (hub_radius + exterior_radius) / 2.
    wingspan = (exterior_radius - radius_bar) * 2.

    # value digitized from fig 14
    thrust = -7.3774 #N, negative because nhat = -zhat_heli

    # values determined from BEM reconstruction
    moment_180_to_0 = 1.086e-1 #Nm
    moment_270_to_90 = 8.707e-3 #Nm

    zhat_heli = cas.cos(tilt) * vect_op.zhat_dm() + cas.sin(tilt) * vect_op.xhat_dm()
    yhat_heli = vect_op.normed_cross(zhat_heli, vect_op.xhat_dm())
    xhat_heli = vect_op.normed_cross(yhat_heli, zhat_heli)
    moment_vector = moment_180_to_0 * xhat_heli + moment_270_to_90 * yhat_heli

    n_hat = -zhat_heli
    u_hat = vect_op.xhat_dm()
    dcm_act, dcm_wind = actuator_flow.construct_actuator_and_wind_reference_frames_from_n_and_u(n_hat, u_hat)
    y_rotor_hat = dcm_act[:, 1]
    z_rotor_hat = dcm_act[:, 2]

    moment_y = cas.mtimes(moment_vector.T, y_rotor_hat)
    moment_z = cas.mtimes(moment_vector.T, z_rotor_hat)

    gamma_yaw = vect_op.angle_between(u_hat, n_hat, z_rotor_hat)

    approx_avg_normal_ind_velocity = 3.0 #in nhat-direction, so positive a = 'down'
    approx_a0 = approx_avg_normal_ind_velocity / u_zero

    c_all, c_numer, c_denom = actuator_coeff.assemble_c_components_from_pieces(thrust, moment_y, moment_z, actuator_dynamic_pressure, area, radius_bar, wingspan)

    veri_test_table = {
        'name': 'helicopter',
        'TiltAngle': tilt * 180./np.pi,
        'nhat': n_hat,
        'uhat': u_hat,
        'RotorMisalignmentAngle': gamma_yaw * 180./np.pi,
        'CT': c_all[0],
        'Cmy': c_all[1],
        'Cmz': c_all[2],
        'ExteriorRadius': exterior_radius,
        'AirDensity': rho,
        'RotorApparentSpeed': u_zero
    }
    print_op.print_dict_as_table(veri_test_table)

    step_history = {0: {'a0': approx_a0, 'acos': 0, 'asin': 0}}
    for step in range(20):
        corr = actuator_flow.get_corr_val_glauert(approx_a0, gamma_yaw)
        chi = actuator_flow.get_wake_angle_chi_jimenez(gamma_yaw, thrust, area, actuator_dynamic_pressure)
        # chi = actuator_flow.get_wake_angle_chi_coleman(gamma_yaw, approx_a0)
        LL_matr = actuator_coeff.get_LL_matrix_from_corr_and_chi(corr, chi)

        a_all = cas.mtimes(LL_matr, c_all)
        a_dict = actuator_flow.unstack_a_all(a_all)
        step_history[step] = a_dict

        approx_a0 = a_dict['a0']

    expected_max_ui = 7.5
    expected_min_ui = -1.5

    all_u = []
    mu = 1.
    for psi in np.linspace(0., 2. * np.pi, 361):
        radius = mu * exterior_radius
        radial_vector = actuator_geom.assemble_radial_vector(radius, y_rotor_hat, z_rotor_hat, psi, for_kite_motion_or_induction_distribution='induction_distribution')
        a_local = actuator_flow.assemble_asym_local_induction_factor(a_dict, psi, mu)
        ui = u_zero * (-1.) * a_local
        all_u += [ui]

    found_max_ui = np.max(np.array(all_u))
    found_min_ui = np.min(np.array(all_u))

    if (found_max_ui - expected_max_ui)**2. > thresh**2.:
        message = 'helicopter qasym found max ui of ' + str(found_max_ui) + ' m/s when ' + str(expected_max_ui) + ' m/s was expected'
        print_op.log_and_raise_error(message)

    if (found_min_ui - expected_min_ui)**2. > thresh**2.:
        message = 'helicopter qasym found min ui of ' + str(found_max_ui) + ' m/s when ' + str(expected_min_ui) + ' m/s was expected'
        print_op.log_and_raise_error(message)

    # print_op.print_dict_as_table(step_history)

    plot_induction_contour(a_dict, u_zero, exterior_radius, y_rotor_hat, z_rotor_hat, 'qasym_helicopter')

    return None



def plot_induction_contour(a_dict, u_zero, exterior_radius, y_rotor_hat, z_rotor_hat, name_string, cmap='RdYlBu_r'):
    dict = {}
    idx = 0
    for mu in np.linspace(0., 1., 30):
        for psi in np.linspace(0., 2. * np.pi, 361):

            radius = mu * exterior_radius
            radial_vector = actuator_geom.assemble_radial_vector(radius, y_rotor_hat, z_rotor_hat, psi, for_kite_motion_or_induction_distribution='induction_distribution')
            a_local = actuator_flow.assemble_asym_local_induction_factor(a_dict, psi, mu)
            ui = u_zero * (-1.) * a_local # * n_hat

            local_dict = {}
            local_dict['mu'] = float(mu)
            local_dict['psi'] = float(psi)
            local_dict['y'] = float(cas.mtimes(radial_vector.T, y_rotor_hat) / exterior_radius)
            local_dict['z'] = float(cas.mtimes(radial_vector.T, z_rotor_hat) / exterior_radius)
            local_dict['a'] = float(a_local)
            local_dict['ui'] = float(ui)
            dict[idx] = local_dict
            idx += 1

    y_array = np.array([local_dict['y'] for local_dict in dict.values()])
    z_array = np.array([local_dict['z'] for local_dict in dict.values()])
    a_array = np.array([local_dict['a'] for local_dict in dict.values()])

    mu_array = np.array([local_dict['mu'] for local_dict in dict.values()])
    psi_array = np.array([local_dict['psi'] for local_dict in dict.values()])
    ui_array = np.array([local_dict['ui'] for local_dict in dict.values()])

    import os
    import matplotlib
    if "MPLBACKEND" not in os.environ:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    base_name = "actuator_q_asym_verification_" + name_string



    def format_ax(ax):
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        ax.xaxis.set_inverted(True)  # because yhatRotor points left in default view
        ax.set_aspect('equal')
        # Draw Y-axis arrow (horizontal, dx=0.5)
        ax.annotate("", xy=(0.5, 0), xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.7})
        # Draw Z-axis arrow (vertical, dy=0.5)
        ax.annotate("", xy=(0, 0.5), xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.7})

    fig, ax = plt.subplots()
    contour = ax.tricontourf(y_array, z_array, a_array, levels=100, cmap=cmap) #, vmin=-6, vmax=-2.5)
    plt.tricontour(y_array, z_array, a_array, levels=100, cmap=cmap)
    ax.tricontour(y_array, z_array, a_array, levels=[a_dict['a0']], colors='black', linewidths=1., linestyles='--', zorder=10)
    fig.colorbar(contour, ax=ax, label='Normal-direction induction factor [-]')
    format_ax(ax)
    plt.savefig(base_name + "_a.pdf", format="pdf")

    fig, ax = plt.subplots()
    contour = ax.tricontourf(y_array, z_array, ui_array, levels=100, cmap=cmap) #, vmin=-6, vmax=-2.5)
    plt.tricontour(y_array, z_array, ui_array, levels=100, cmap=cmap)
    ax.tricontour(y_array, z_array, ui_array, levels=[u_zero * (-1.) * a_dict['a0']], colors='black', linewidths=1., linestyles='--', zorder=10)
    fig.colorbar(contour, ax=ax, label='Normal-direction induced velocity [m/s]')
    format_ax(ax)
    plt.savefig(base_name + "_ui.pdf", format="pdf")

    fig, ax = plt.subplots()
    contour = ax.tricontourf(y_array, z_array, - a_array, levels=100, cmap=cmap) #, vmin=-6, vmax=-2.5)
    plt.tricontour(y_array, z_array, - a_array, levels=100, cmap=cmap)
    ax.tricontour(y_array, z_array, - a_array, levels=[(-1.) * a_dict['a0']], colors='black', linewidths=1., linestyles='--', zorder=10)
    fig.colorbar(contour, ax=ax, label='Normal-direction induced velocity / u0 [-]')
    format_ax(ax)
    plt.savefig(base_name + "_neg_a.pdf", format="pdf")

    plt.show()
    return None


def pitch_test_oc6_35():

    # https://re.public.polimi.it/retrieve/f3184f82-aed8-4c8b-9f4a-de090f07bd44/199-RENE24-FOWTALM.pdf
    # https://wdh.energy.gov/ds/oc6/oc6.phase3
    # https://docs.nlr.gov/docs/fy23osti/83102.pdf

    import pandas as pd
    df = pd.read_csv(
        # "oc6.phase3.dtu1.m53.lc35.forces.txt",
        # sep=r"\s+",
        # header=None,
        "oc6.phase3.sanvito_reconstructed.lc35.forces.txt",
        sep=",",
        header=0,
        names=["time", "pitch_deg", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    )

    n_periods = 3
    period = df["time"].iloc[-1] - df["time"].iloc[0]
    df_stacked = pd.concat(
        [
            df.assign(time=df["time"] + i * period).iloc[1 if i > 0 else 0:]
            for i in range(n_periods)
        ],
        ignore_index=True
    )

    xhat = vect_op.xhat_dm()
    yhat = vect_op.yhat_dm()
    zhat = vect_op.zhat_dm()

    rho = 1.177 # kg/m^3, pg 12, https://docs.nlr.gov/docs/fy23osti/83102.pdf
    u_infty = 4.0 #m/s, % table 2, pg. 3, Sanvito
    # see page 19 https://docs.nlr.gov/docs/fy23osti/83102.pdf
    # 4.00 m/s is wind speed measured in the tunnel.
    # 4.19 m/s is equivalent wind-speed after correcting for blockage.

    # from table A.6, https://docs.nlr.gov/docs/fy23osti/83102.pdf
    freq_pitch_hz = 1.0  # 1/s
    amp_pitch_deg = 1.4  # deg
    rotor_rpm = 240.  # rpm
    blade_pitch_deg = 0.  # deg

    phase_shift_pitch = 0.  # pg. 3 https://re.public.polimi.it/retrieve/f3184f82-aed8-4c8b-9f4a-de090f07bd44/199-RENE24-FOWTALM.pdf
    phase_shift_azimuth_deg = 7.  # pg. 3 Sanvito
    tower_tilt_rad = 5. * np.pi / 180  # pg 11, https://docs.nlr.gov/docs/fy23osti/83102.pdf

    rotor_rad_p_s = rotor_rpm * 2. * np.pi / 60.
    amp_surge_rad = amp_pitch_deg * np.pi / 180.

    # from Table 2, https://docs.nlr.gov/docs/fy23osti/83102.pdf
    rotor_diameter = 2.38132  # m,
    hub_diameter = 0.178  # m
    tower_base_offset = 0.730  # m
    tower_length = 1.400  # m
    twr2shft = 0.064  # m
    rotor_overhang = 0.139  # m

    exterior_radius = rotor_diameter / 2.
    hub_radius = hub_diameter / 2.
    wingspan = (rotor_diameter - hub_diameter) / 2.
    radius_bar = exterior_radius - wingspan / 2.
    area = np.pi * exterior_radius**2.

    time_sym = cas.SX.sym('time_sym')
    mu_sym = cas.SX.sym('mu_sym')
    a_all_sym = cas.SX.sym('a_all_sym', (3, 1))
    keller_constant_sym = cas.SX.sym('keller_constant_sym')
    force_vec_sym = cas.SX.sym('force_vec_sym', (3, 1))
    moment_vec_sym = cas.SX.sym('moment_vec_sym', (3, 1))

    mu_eval = np.linspace(0., 1., 100)

    # table 3, pg. 9 https://docs.nlr.gov/docs/fy23osti/83102.pdf
    mu_twist = np.array([0.075, 0.124, 0.189, 0.258, 0.327, 0.397, 0.466, 0.533, 0.596, 0.656, 0.711, 0.761, 0.807, 0.847, 0.882, 0.913, 0.940, 0.963, 0.983, 1.000])
    twist_deg = np.array([17.07668, 17.04199, 15.77593, 12.30509, 9.98299, 8.65143, 7.56522, 6.38165, 5.08008, 3.79042, 2.61685, 1.59090, 0.71754, 0.03751, -0.53510, -1.03393, -1.46251, -1.61172, -1.60710, -1.72236])
    twist_at_mu_deg_interp = np.interp(mu_eval, mu_twist, twist_deg)

    def get_scalar_beta_and_omega_pitch(time):
        amp_pitch_rad = amp_pitch_deg * np.pi / 180.
        omega_pitch_frequency = 2. * np.pi * freq_pitch_hz
        beta_pitch_rad = amp_pitch_rad * cas.sin(omega_pitch_frequency * time + phase_shift_pitch)
        omega_pitch_rad_p_s = cas.jacobian(beta_pitch_rad, time)
        return beta_pitch_rad, omega_pitch_rad_p_s
    beta_pitch_rad_sym, omega_pitch_rad_p_s_sym = get_scalar_beta_and_omega_pitch(time_sym)
    pitch_rad_fun = cas.Function('pitch_rad_fun', [time_sym], [beta_pitch_rad_sym])

    def get_base_dcms(time):
        xhat_cs1 = vect_op.xhat_dm()
        zhat_cs1 = vect_op.zhat_dm()
        yhat_cs1 = vect_op.yhat_dm()
        dcm_cs1 = cas.horzcat(xhat_cs1, yhat_cs1, zhat_cs1)

        beta_pitch_rad, _ = get_scalar_beta_and_omega_pitch(time)
        tower_angle_rad = tower_tilt_rad - beta_pitch_rad
        zhat_cs2 = (cas.cos(tower_angle_rad) * zhat_cs1 - cas.sin(tower_angle_rad) * xhat_cs1)
        yhat_cs2 = -1. * yhat_cs1
        xhat_cs2 = vect_op.normed_cross(yhat_cs2, zhat_cs2)
        dcm_cs2 = cas.horzcat(xhat_cs2, yhat_cs2, zhat_cs2)

        xhat_cs3 = -cas.cos(tower_tilt_rad) * xhat_cs2 - cas.sin(tower_tilt_rad) * zhat_cs2
        yhat_cs3 = yhat_cs1
        zhat_cs3 = vect_op.normed_cross(xhat_cs3, yhat_cs3)
        dcm_cs3 = cas.horzcat(xhat_cs3, yhat_cs3, zhat_cs3)

        return dcm_cs1, dcm_cs2, dcm_cs3
    _, _, dcm_cs3_sym = get_base_dcms(time_sym)
    dcm_cs3_fun = cas.Function('dcm_cs3_fun', [time_sym], [dcm_cs3_sym])

    def get_cs_center_positions(time):
        dcm_cs1, dcm_cs2, dcm_cs3 = get_base_dcms(time)
        zhat_cs2 = dcm_cs2[:, 2]
        xhat_cs3 = dcm_cs3[:, 0]
        center_cs1_wrt_earth = tower_base_offset * zhat
        center_cs2_wrt_earth = center_cs1_wrt_earth + tower_length * zhat_cs2
        center_cs3_wrt_earth = center_cs2_wrt_earth + twr2shft * zhat_cs2 - rotor_overhang * xhat_cs3
        return center_cs1_wrt_earth, center_cs2_wrt_earth, center_cs3_wrt_earth

    def get_scalar_psi_and_omega_rot(bdx, time):
        omega_rot_rad_p_s = rotor_rpm * 2. * np.pi / 60.
        psi_0 = phase_shift_azimuth_deg * np.pi / 180.
        psi_bladeshift = bdx * 2. * np.pi / 3.
        psi_understood = psi_0 + psi_bladeshift + omega_rot_rad_p_s * time
        return psi_understood, omega_rot_rad_p_s

    def get_understood_order_of_cylindrical_coordinates():
        # bladeRefVels are mostly u_infty at y/1 position -> nhat
        # bladeForces are zero at z/2 position -> rhat
        return {'that': 0, 'nhat': 1, 'rhat': 2}
    def restack_cylindrical_coordinates(ehats_in_dict):
        coord_order = get_understood_order_of_cylindrical_coordinates()
        dcm_restacked = cas.DM.zeros((3, 0))
        for dir, ddx in coord_order.items():
            if ddx == dcm_restacked.shape[1]:
                dcm_restacked = cas.horzcat(dcm_restacked, ehats_in_dict[dir])
        return dcm_restacked

    def get_blade_fixed_dcms(bdx, time):
        cylindrical_coord_order = get_understood_order_of_cylindrical_coordinates()
        base_dcm = cas.DM.eye(3)
        nhat_wrt_blade = base_dcm[:, cylindrical_coord_order['nhat']]
        rhat_wrt_blade = base_dcm[:, cylindrical_coord_order['rhat']]
        that_wrt_blade = base_dcm[:, cylindrical_coord_order['that']]
        ehat_dict_wrt_blade = {'nhat': nhat_wrt_blade, 'rhat': rhat_wrt_blade, 'that': that_wrt_blade}
        dcm_blade_wrt_blade = restack_cylindrical_coordinates(ehat_dict_wrt_blade)

        psi, _ = get_scalar_psi_and_omega_rot(bdx, time)

        import copy
        xhat_wrt_rotor = copy.deepcopy(xhat)
        yhat_wrt_rotor = copy.deepcopy(yhat)
        zhat_wrt_rotor = copy.deepcopy(zhat)

        nhat_wrt_rotor = xhat_wrt_rotor
        rhat_wrt_rotor = actuator_geom.parametric_rhat(yhat_wrt_rotor, zhat_wrt_rotor, psi)
        that_wrt_rotor = vect_op.normed_cross(nhat_wrt_rotor, rhat_wrt_rotor)
        ehat_dict_wrt_rotor = {'nhat': nhat_wrt_rotor, 'rhat': rhat_wrt_rotor, 'that': that_wrt_rotor}
        dcm_blade_wrt_rotor = restack_cylindrical_coordinates(ehat_dict_wrt_rotor)

        _, _, dcm_cs3 = get_base_dcms(time)
        dcm_blade_wrt_earth = cas.mtimes(dcm_cs3, dcm_blade_wrt_rotor)

        return dcm_blade_wrt_blade, dcm_blade_wrt_rotor, dcm_blade_wrt_earth

    def get_rotation_arm_at_station(bdx, time, mu):
        _, _, dcm_blade_wrt_earth = get_blade_fixed_dcms(bdx, time)
        q_center_cs1, _, q_center_cs3 = get_cs_center_positions(time)
        local_radius = exterior_radius * mu
        coord_order = get_understood_order_of_cylindrical_coordinates()
        rhat_blade_wrt_earth = dcm_blade_wrt_earth[:, coord_order['rhat']]
        q_local = q_center_cs3 + rhat_blade_wrt_earth * local_radius
        rotation_arm = q_local - q_center_cs3
        return q_local, rotation_arm

    def get_dq_blade_at_station_wrt_earth(bdx, time, mu):

        _, omega_pitch_rad_p_s = get_scalar_beta_and_omega_pitch(time)
        _, omega_rot_rad_p_s = get_scalar_psi_and_omega_rot(bdx, time)

        dcm_cs1, _, dcm_cs3 = get_base_dcms(time)

        q_center_cs1, _, q_center_cs3 = get_cs_center_positions(time)
        q_local, rotation_arm = get_rotation_arm_at_station(bdx, time, mu)
        xhat_cs3 = dcm_cs3[:, 0]
        omega_rot_vec = omega_rot_rad_p_s * xhat_cs3
        rotation_motion = vect_op.cross(omega_rot_vec, rotation_arm)

        pitching_arm = q_local - q_center_cs1
        yhat_cs1 = dcm_cs1[:, 1]
        omega_pitch_vec = omega_pitch_rad_p_s * yhat_cs1
        pitching_motion = vect_op.cross(omega_pitch_vec, pitching_arm)

        dq_blade = rotation_motion + pitching_motion
        return dq_blade

    def get_dcm_act_and_wind(time):
        _, _, center_cs3 = get_cs_center_positions(time)
        dq_actuator = cas.jacobian(center_cs3, time)

        _, _, dcm_cs3 = get_base_dcms(time)
        xhat_cs3 = dcm_cs3[:, 0]

        u_zero = u_infty * vect_op.xhat_dm() - dq_actuator
        u_hat = vect_op.normalize(u_zero)
        n_hat = xhat_cs3

        dcm_act, dcm_wind = actuator_flow.construct_actuator_and_wind_reference_frames_from_n_and_u(n_hat, u_hat)
        z_rotor_hat = dcm_act[:, 2]

        gamma_yaw = vect_op.angle_between(u_hat, n_hat, z_rotor_hat)
        return u_zero, dcm_act, dcm_wind, gamma_yaw

    def get_u_eff_at_station_wrt_blade(bdx, a_all, time, mu):
        u_zero, dcm_act, dcm_wind, gamma_yaw = get_dcm_act_and_wind(time)
        n_hat = dcm_act[:, 0]
        y_rotor_hat = dcm_act[:, 1]
        z_rotor_hat = dcm_act[:, 2]

        _, _, dcm_cs3 = get_base_dcms(time)
        yhat_cs3 = dcm_cs3[:, 1]
        zhat_cs3 = dcm_cs3[:, 2]
        psi_blade, _ = get_scalar_psi_and_omega_rot(bdx, time)

        r_hat_blade = actuator_geom.parametric_rhat(yhat_cs3, zhat_cs3, psi_blade)
        psi_induction = cas.atan2(-cas.dot(r_hat_blade, y_rotor_hat), cas.dot(r_hat_blade, z_rotor_hat))

        a_local = actuator_flow.assemble_asym_local_induction_factor(actuator_flow.unstack_a_all(a_all),
                                                                     psi_induction, mu)
        u_induced = vect_op.norm(u_zero) * a_local * (-1.) * n_hat
        vec_u_ref_wrt_earth = u_infty * xhat + u_induced
        dq_blade_wrt_earth = get_dq_blade_at_station_wrt_earth(bdx, time, mu)

        dcm_blade_wrt_blade, _, dcm_blade_wrt_earth = get_blade_fixed_dcms(bdx, time)

        vec_u_eff_wrt_earth = vec_u_ref_wrt_earth - dq_blade_wrt_earth
        vec_u_eff_in_cylindrical_coords = cas.mtimes(dcm_blade_wrt_earth.T, vec_u_eff_wrt_earth)
        vec_u_eff_wrt_blade = cas.mtimes(dcm_blade_wrt_blade, vec_u_eff_in_cylindrical_coords)
        return vec_u_eff_wrt_blade

    def get_flow_angle_at_station(bdx, a_all, time, mu):
        vec_u_eff_wrt_blade = get_u_eff_at_station_wrt_blade(bdx, a_all, time, mu)

        cylindrical_coord_order = get_understood_order_of_cylindrical_coordinates()

        u_eff_nhat = vec_u_eff_wrt_blade[cylindrical_coord_order['nhat']]
        u_eff_that = vec_u_eff_wrt_blade[cylindrical_coord_order['that']]

        flow_angle_understood_rad = cas.atan2(u_eff_nhat, -u_eff_that)
        flow_angle_deg = flow_angle_understood_rad * 180. / np.pi
        return flow_angle_deg
    eff_flow_angle_deg_fun = cas.Function('eff_flow_angle_deg_fun', [a_all_sym, time_sym, mu_sym],
                                  [get_flow_angle_at_station(0, a_all_sym, time_sym, mu_sym)])
    app_flow_angle_deg_fun = cas.Function('app_flow_angle_deg_fun', [time_sym, mu_sym],
                                  [get_flow_angle_at_station(0, cas.DM.zeros((3, 1)), time_sym, mu_sym)])

    def get_normal_flow_velocity_at_station(bdx, a_all, time, mu):
        vec_u_eff_wrt_blade = get_u_eff_at_station_wrt_blade(bdx, a_all, time, mu)
        cylindrical_coord_order = get_understood_order_of_cylindrical_coordinates()
        u_eff_nhat = vec_u_eff_wrt_blade[cylindrical_coord_order['nhat']]
        return u_eff_nhat
    u_eff_nhat_fun = cas.Function('u_eff_nhat_fun', [a_all_sym, time_sym, mu_sym], [get_normal_flow_velocity_at_station(0, a_all_sym, time_sym, mu_sym)])
    u_app_nhat_fun = cas.Function('u_eff_nhat_fun', [time_sym, mu_sym], [get_normal_flow_velocity_at_station(0, cas.DM.zeros((3, 1)), time_sym, mu_sym)])


    def calculate_adot(time, a_all, force_vec, moment_vec, keller_constant):
        u_zero, dcm_act, dcm_wind, gamma_yaw = get_dcm_act_and_wind(time)
        n_hat = dcm_act[:, 0]
        y_rotor_hat = dcm_act[:, 1]
        z_rotor_hat = dcm_act[:, 2]

        t_star = exterior_radius / vect_op.norm(u_zero)
        actuator_dynamic_pressure = 0.5 * rho * cas.mtimes(u_zero.T, u_zero)

        thrust = cas.mtimes(n_hat.T, force_vec)
        moment_y = cas.mtimes(y_rotor_hat.T, moment_vec)
        moment_z = cas.mtimes(z_rotor_hat.T, moment_vec)

        c_all, c_numer, c_denom = actuator_coeff.assemble_c_components_from_pieces(thrust, moment_y, moment_z,
                                                                                   actuator_dynamic_pressure, area,
                                                                                   radius_bar, wingspan)
        MM = actuator_coeff.get_MM_matrix()

        approx_a0 = a_all[0]

        corr = actuator_flow.get_corr_val_glauert(approx_a0, gamma_yaw)
        chi = actuator_flow.get_wake_angle_chi_jimenez(gamma_yaw, thrust, area, actuator_dynamic_pressure)
        LLinv, _, _ = actuator_coeff.get_LL_inv_matrix_from_corr_and_chi(corr, chi)

        positive_rotation_about_xhat = True
        dot_dcm_act = cas.reshape(cas.jacobian(cas.reshape(dcm_act, (9, 1)), time), (3, 3))
        rhs_keller = actuator_coeff.make_keller_forcing_term(t_star, LLinv, keller_constant, positive_rotation_about_xhat,
                                                         dcm_act, dot_dcm_act)

        term_2 = cas.mtimes(LLinv, a_all)
        adot_all = cas.mtimes(cas.inv(MM), (c_all - term_2 + rhs_keller)) / t_star
        return adot_all
    adot_all_fun = cas.Function('adot_all_fun', [time_sym, a_all_sym, force_vec_sym, moment_vec_sym, keller_constant_sym], [calculate_adot(time_sym, a_all_sym, force_vec_sym, moment_vec_sym, keller_constant_sym)])

    keller_step_history = {}
    keller_time_history = {}
    keller_nhat_dot_xhat_history = {}
    keller_pitch_history = {}
    keller_alpha_deg_history = {}
    keller_u_eff_nhat_history = {}
    keller_plot_style = {-2.0:':', 0.:'-', 1.0:'--', 2.0:'-.'}
    for keller_constant in keller_plot_style.keys():

        step_history = {0: {'a0': 0, 'acos': 0, 'asin': 0}}
        time_history = {0: 0}
        nhat_dot_xhat_history = {}
        pitch_history = {}
        alpha_deg_history = {}
        u_eff_nhat_history = {}
        alpha_app_deg_history = {}
        u_app_nhat_history = {}
        for idx in range(len(df_stacked)):
            time_s = df_stacked["time"][idx]
            fx_read = df_stacked["Fx"][idx]
            fy_read = df_stacked["Fy"][idx]
            fz_read = df_stacked["Fz"][idx]
            mx_read = df_stacked["Mx"][idx]
            my_read = df_stacked["My"][idx]
            mz_read = df_stacked["Mz"][idx]

            pitch_angle_rad = pitch_rad_fun(time_s)
            if (df_stacked["pitch_deg"][idx] - (pitch_angle_rad * 180./np.pi))**2. > 1e-3:
                message = 'wrong pitch definition'
                print_op.log_and_raise_error(message)

            dcm_cs3 = dcm_cs3_fun(time_s)
            xhat_cs3 = dcm_cs3[:, 0]
            yhat_cs3 = dcm_cs3[:, 1]
            zhat_cs3 = dcm_cs3[:, 2]
            total_aero_force = (fx_read * xhat_cs3 + fy_read * yhat_cs3 + fz_read * zhat_cs3)
            total_aero_moment = (mx_read * xhat_cs3 + my_read * yhat_cs3 + mz_read * zhat_cs3)

            a_all = actuator_flow.restack_a_dict(step_history[idx])
            adot_now = adot_all_fun(time_s, a_all, total_aero_force, total_aero_moment, keller_constant)
            if idx + 1 < len(df_stacked):
                time_next = df_stacked["time"].iloc[idx + 1]
                dt_next = time_next - time_s
                a_next = cas.DM(a_all) + dt_next * adot_now
                step_history[idx + 1] = actuator_flow.unstack_a_all(a_next)

            # Store the time corresponding to the CURRENT velocity calculation
            time_history[idx + 1] = time_s
            nhat_dot_xhat_history[idx + 1] = cas.mtimes(xhat_cs3.T, vect_op.xhat_dm())
            pitch_history[idx + 1] = pitch_angle_rad * 180. / np.pi

            store_u_eff_nhat = []
            store_flow_angle_deg = []
            store_u_app_nhat = []
            store_app_flow_angle_deg = []
            for mu in mu_eval:
                store_u_eff_nhat += [float(u_eff_nhat_fun(a_all, time_s, mu))]
                store_flow_angle_deg += [float(eff_flow_angle_deg_fun(a_all, time_s, mu))]
                store_u_app_nhat += [float(u_app_nhat_fun(time_s, mu))]
                store_app_flow_angle_deg += [float(app_flow_angle_deg_fun(time_s, mu))]
            store_alpha_deg = np.array(store_flow_angle_deg) - twist_at_mu_deg_interp
            store_alpha_app_deg = np.array(store_app_flow_angle_deg) - twist_at_mu_deg_interp

            u_eff_nhat_history[idx + 1] = np.array(store_u_eff_nhat)
            alpha_deg_history[idx + 1] = np.array(store_alpha_deg)
            u_app_nhat_history[idx + 1] = np.array(store_u_app_nhat)
            alpha_app_deg_history[idx + 1] = np.array(store_alpha_app_deg)

        keller_step_history[keller_constant] = step_history
        keller_time_history[keller_constant] = time_history
        keller_nhat_dot_xhat_history[keller_constant] = nhat_dot_xhat_history
        keller_pitch_history[keller_constant] = pitch_history
        keller_alpha_deg_history[keller_constant] = alpha_deg_history
        keller_u_eff_nhat_history[keller_constant] = u_eff_nhat_history

    import os
    import matplotlib
    if "MPLBACKEND" not in os.environ:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    for keller_constant, keller_linestyle in keller_plot_style.items():
        plt.plot(
            np.array([float(time) for time in keller_time_history[keller_constant].values()])[1:],
            np.array([float(pitch) for pitch in keller_pitch_history[keller_constant].values()]),
            linestyle=keller_linestyle, color='k'
        )
    plt.xlabel('time [s]')
    plt.ylabel('pitch angle [deg]')

    alpha_mu_sanvito = [0.15821, 0.16247, 0.16673, 0.17099, 0.17526, 0.17952, 0.18378, 0.18804, 0.1923, 0.19656,
                        0.20082, 0.20509, 0.20935, 0.21361, 0.21787, 0.22213, 0.22639, 0.23066, 0.23492, 0.23918,
                        0.24344, 0.2477, 0.25196, 0.25623, 0.26049, 0.26475, 0.26901, 0.27327, 0.27753, 0.28179,
                        0.28606, 0.29032, 0.29458, 0.29884, 0.3031, 0.30736, 0.31163, 0.31589, 0.32015, 0.32441,
                        0.32867, 0.33293, 0.33719, 0.34146, 0.34572, 0.34998, 0.35424, 0.3585, 0.36276, 0.36703,
                        0.37129, 0.37555, 0.37981, 0.38407, 0.38833, 0.39259, 0.39686, 0.40112, 0.40538, 0.40964,
                        0.4139, 0.41816, 0.42243, 0.42669, 0.43095, 0.43521, 0.43947, 0.44373, 0.44799, 0.45226,
                        0.45652, 0.46078, 0.46504, 0.4693, 0.47356, 0.47783, 0.48209, 0.48635, 0.49061, 0.49487,
                        0.49913, 0.5034, 0.50766, 0.51192, 0.51618, 0.52044, 0.5247, 0.52896, 0.53323, 0.53749, 0.54175,
                        0.54601, 0.55027, 0.55453, 0.5588, 0.56306, 0.56732, 0.57158, 0.57584, 0.5801, 0.58436, 0.58863,
                        0.59289, 0.59715, 0.60141, 0.60567, 0.60993, 0.6142, 0.61846, 0.62272, 0.62698, 0.63124, 0.6355,
                        0.63976, 0.64403, 0.64829, 0.65255, 0.65681, 0.66107, 0.66533, 0.6696, 0.67386, 0.67812,
                        0.68238, 0.68664, 0.6909, 0.69517, 0.69943, 0.70369, 0.70795, 0.71221, 0.71647, 0.72073, 0.725,
                        0.72926, 0.73352, 0.73778, 0.74204, 0.7463, 0.75057, 0.75483, 0.75909, 0.76335, 0.76761,
                        0.77187, 0.77613, 0.7804, 0.78466, 0.78892, 0.79318, 0.79744, 0.8017, 0.80597, 0.81023, 0.81449,
                        0.81875, 0.82301, 0.82727, 0.83153, 0.8358, 0.84006, 0.84432, 0.84858, 0.85284, 0.8571, 0.86137,
                        0.86563, 0.86989, 0.87415, 0.87841, 0.88267, 0.88693, 0.8912, 0.89546, 0.89972, 0.90398,
                        0.90824, 0.9125, 0.91677, 0.92103, 0.92529, 0.92955, 0.93381, 0.93807, 0.94234, 0.9466, 0.95086,
                        0.95512, 0.95938, 0.96364, 0.9679, 0.97217, 0.97643, 0.98069, 0.98495, 0.98921]
    alpha_t0_sanvito = [19.9577, 19.2315, 18.5067, 17.7848, 17.0672, 16.3556, 15.6518, 14.9576, 14.2752, 13.6073,
                        12.9569, 12.3277, 11.7243, 11.1527, 10.6204, 10.1351, 9.7013, 9.32, 8.9874, 8.6971, 8.4412,
                        8.2122, 8.0034, 7.8095, 7.6263, 7.4508, 7.2806, 7.1152, 6.9546, 6.7989, 6.6479, 6.5016, 6.3599,
                        6.2228, 6.0902, 5.9618, 5.8378, 5.7179, 5.6021, 5.4903, 5.3824, 5.2783, 5.178, 5.0814, 4.9884,
                        4.899, 4.813, 4.7304, 4.6511, 4.5751, 4.5024, 4.4328, 4.3663, 4.3029, 4.2425, 4.1848, 4.1299,
                        4.0775, 4.0276, 3.9801, 3.9349, 3.8918, 3.8509, 3.8121, 3.7752, 3.7403, 3.7073, 3.6761, 3.6468,
                        3.6192, 3.5933, 3.5691, 3.5465, 3.5256, 3.5062, 3.4885, 3.4722, 3.4575, 3.4443, 3.4326, 3.4223,
                        3.4136, 3.4062, 3.4003, 3.3958, 3.3927, 3.391, 3.3906, 3.3917, 3.3942, 3.398, 3.4032, 3.4098,
                        3.4177, 3.4271, 3.4378, 3.4498, 3.4633, 3.4781, 3.4943, 3.5119, 3.5308, 3.5509, 3.5723, 3.595,
                        3.6188, 3.6437, 3.6698, 3.6969, 3.7252, 3.7544, 3.7847, 3.816, 3.8483, 3.8814, 3.9156, 3.9506,
                        3.9865, 4.0232, 4.0608, 4.0992, 4.1384, 4.1784, 4.2191, 4.2606, 4.3028, 4.3457, 4.3893, 4.4335,
                        4.4784, 4.5239, 4.57, 4.6166, 4.6638, 4.7115, 4.7597, 4.8083, 4.8573, 4.9066, 4.9563, 5.0061,
                        5.0561, 5.1061, 5.1562, 5.2063, 5.2564, 5.3066, 5.3568, 5.407, 5.4573, 5.5077, 5.5582, 5.6087,
                        5.6593, 5.71, 5.7607, 5.8115, 5.8624, 5.9132, 5.964, 6.0149, 6.0657, 6.1165, 6.1674, 6.2183,
                        6.2693, 6.3203, 6.3715, 6.4227, 6.4741, 6.5257, 6.5775, 6.6295, 6.682, 6.735, 6.7889, 6.844,
                        6.9005, 6.9584, 7.0163, 7.0722, 7.1255, 7.1768, 7.2265, 7.2747, 7.3213, 7.3658, 7.4079, 7.4464,
                        7.4796, 7.5062, 7.525, 7.5366, 7.5454, 7.5577, 7.5775]
    alpha_t1_sanvito = [21.6555, 20.8665, 20.0658, 19.2641, 18.4721, 17.6996, 16.9547, 16.2432, 15.5689, 14.9334,
                        14.3373, 13.7796, 13.2589, 12.7733, 12.3208, 11.899, 11.5051, 11.136, 10.7894, 10.4629, 10.155,
                        9.8638, 9.5882, 9.3269, 9.0789, 8.8432, 8.6189, 8.4055, 8.2021, 8.0081, 7.8231, 7.6464, 7.4778,
                        7.3166, 7.1626, 7.0154, 6.8747, 6.7401, 6.6113, 6.488, 6.3698, 6.2565, 6.1478, 6.0434, 5.9432,
                        5.847, 5.7545, 5.6656, 5.5802, 5.4982, 5.4194, 5.3437, 5.2709, 5.2011, 5.1341, 5.0698, 5.0082,
                        4.9492, 4.8926, 4.8386, 4.7869, 4.7375, 4.6905, 4.6457, 4.6031, 4.5626, 4.5243, 4.4881, 4.454,
                        4.4219, 4.3918, 4.3637, 4.3375, 4.3132, 4.2906, 4.2698, 4.2507, 4.2332, 4.2172, 4.2028, 4.1899,
                        4.1785, 4.1684, 4.1597, 4.1524, 4.1464, 4.1416, 4.1381, 4.1358, 4.1347, 4.1348, 4.136, 4.1383,
                        4.1417, 4.1462, 4.1517, 4.1582, 4.1657, 4.1743, 4.1837, 4.194, 4.2053, 4.2173, 4.2305, 4.2449,
                        4.2611, 4.2793, 4.2999, 4.3232, 4.3494, 4.3784, 4.4101, 4.444, 4.4799, 4.5172, 4.5556, 4.5948,
                        4.6346, 4.6747, 4.7151, 4.7555, 4.7961, 4.8368, 4.8775, 4.9184, 4.9594, 5.0005, 5.0417, 5.0831,
                        5.1245, 5.1661, 5.2079, 5.2497, 5.2918, 5.3339, 5.3762, 5.4187, 5.4613, 5.5041, 5.5471, 5.5902,
                        5.6336, 5.6771, 5.7209, 5.7648, 5.809, 5.8535, 5.8982, 5.9432, 5.9885, 6.0341, 6.0801, 6.1264,
                        6.1731, 6.2202, 6.2677, 6.3157, 6.3641, 6.4129, 6.4621, 6.5118, 6.5619, 6.6124, 6.6634, 6.7147,
                        6.7665, 6.8187, 6.8713, 6.9243, 6.9777, 7.0315, 7.0857, 7.1403, 7.1951, 7.2503, 7.3058, 7.3614,
                        7.4171, 7.4728, 7.5282, 7.5832, 7.6372, 7.6898, 7.7396, 7.785, 7.8246, 7.8587, 7.8895, 7.9184,
                        7.9461, 7.9729, 7.9991, 8.0248, 8.0503, 8.0755, 8.1006]
    alpha_t2_sanvito = [23.4142, 22.7495, 22.0371, 21.2357, 20.2496, 19.1772, 18.3097, 17.5659, 16.8854, 16.2532,
                        15.664, 15.1138, 14.5992, 14.1179, 13.6677, 13.2468, 12.8538, 12.4863, 12.14, 11.8112, 11.4973,
                        11.1964, 10.9068, 10.6274, 10.3572, 10.0954, 9.8413, 9.5945, 9.3545, 9.1211, 8.8942, 8.6746,
                        8.4637, 8.2627, 8.0731, 7.8959, 7.7311, 7.5772, 7.4315, 7.2962, 7.1729, 7.0603, 6.9564, 6.8591,
                        6.7669, 6.6786, 6.5935, 6.5108, 6.4302, 6.3514, 6.2743, 6.1987, 6.1246, 6.052, 5.9808, 5.911,
                        5.8427, 5.7758, 5.7104, 5.6464, 5.584, 5.5232, 5.4641, 5.4066, 5.3509, 5.2972, 5.2453, 5.1956,
                        5.1481, 5.103, 5.0604, 5.0205, 4.9832, 4.9484, 4.9162, 4.8866, 4.8595, 4.8349, 4.8126, 4.7928,
                        4.7754, 4.7602, 4.7472, 4.7365, 4.7278, 4.7212, 4.7165, 4.7137, 4.7126, 4.7132, 4.7153, 4.7189,
                        4.7237, 4.7299, 4.7374, 4.7464, 4.7567, 4.7685, 4.7817, 4.7963, 4.8124, 4.8298, 4.8486, 4.8687,
                        4.89, 4.9125, 4.9361, 4.9608, 4.9864, 5.013, 5.0403, 5.0685, 5.0974, 5.127, 5.1573, 5.1883,
                        5.2199, 5.2522, 5.2851, 5.3187, 5.3529, 5.3877, 5.4232, 5.4593, 5.496, 5.5334, 5.5715, 5.6102,
                        5.6496, 5.6897, 5.7306, 5.7721, 5.8145, 5.8576, 5.9014, 5.9459, 5.9911, 6.0369, 6.0833, 6.1301,
                        6.1774, 6.2251, 6.2731, 6.3214, 6.37, 6.4186, 6.4674, 6.5161, 6.5648, 6.6134, 6.6617, 6.7097,
                        6.7572, 6.8043, 6.8508, 6.8968, 6.9423, 6.9874, 7.0323, 7.0771, 7.1218, 7.1665, 7.2115, 7.2568,
                        7.3025, 7.349, 7.3962, 7.4445, 7.494, 7.545, 7.5978, 7.6525, 7.709, 7.7671, 7.8264, 7.8866,
                        7.9472, 8.008, 8.0683, 8.1276, 8.1854, 8.241, 8.2939, 8.3432, 8.3885, 8.4296, 8.4668, 8.5004,
                        8.5308, 8.5583, 8.5832, 8.606, 8.6271, 8.6468, 8.6656, 8.684]
    alpha_t3_sanvito = [21.8605, 20.9911, 20.034, 19.0704, 18.1881, 17.4082, 16.7136, 16.0795, 15.4812, 14.9053,
                        14.3441, 13.7924, 13.2457, 12.7002, 12.1657, 11.6652, 11.2131, 10.8109, 10.4526, 10.1305,
                        9.8373, 9.5668, 9.3134, 9.073, 8.8431, 8.6223, 8.4092, 8.2032, 8.0037, 7.8103, 7.6228, 7.4412,
                        7.2655, 7.096, 6.9332, 6.7778, 6.6306, 6.4923, 6.3632, 6.2436, 6.1332, 6.0313, 5.9369, 5.8487,
                        5.7649, 5.6842, 5.6051, 5.5271, 5.4503, 5.375, 5.3012, 5.229, 5.1586, 5.0901, 5.0236, 4.9591,
                        4.8967, 4.8367, 4.779, 4.7237, 4.671, 4.6208, 4.5732, 4.528, 4.4851, 4.4444, 4.4058, 4.3693,
                        4.3347, 4.3019, 4.2707, 4.2412, 4.2132, 4.1865, 4.1612, 4.1371, 4.1141, 4.0922, 4.0713, 4.052,
                        4.0343, 4.0187, 4.0052, 3.9939, 3.985, 3.9784, 3.9741, 3.9722, 3.9725, 3.9751, 3.9797, 3.9864,
                        3.9951, 4.0057, 4.018, 4.0322, 4.048, 4.0654, 4.0843, 4.1047, 4.1264, 4.1495, 4.1739, 4.1995,
                        4.2263, 4.2542, 4.2831, 4.3131, 4.3441, 4.3759, 4.4086, 4.442, 4.4762, 4.511, 4.5465, 4.5826,
                        4.6192, 4.6563, 4.694, 4.7321, 4.7707, 4.8098, 4.8493, 4.8891, 4.9294, 4.9701, 5.0111, 5.0524,
                        5.0942, 5.1362, 5.1786, 5.2213, 5.2643, 5.3076, 5.3512, 5.3951, 5.4393, 5.4838, 5.5285, 5.5735,
                        5.6188, 5.6644, 5.7102, 5.7563, 5.8027, 5.8493, 5.8962, 5.9433, 5.9908, 6.0385, 6.0864, 6.1347,
                        6.1832, 6.2321, 6.2812, 6.3307, 6.3804, 6.4305, 6.481, 6.5318, 6.583, 6.6346, 6.6866, 6.7391,
                        6.7919, 6.8452, 6.8989, 6.9531, 7.0077, 7.0628, 7.1183, 7.1742, 7.2305, 7.2873, 7.3443, 7.4016,
                        7.459, 7.5164, 7.5736, 7.6304, 7.6863, 7.741, 7.7938, 7.8441, 7.8911, 7.9342, 7.9734, 8.0084,
                        8.0393, 8.066, 8.0887, 8.1075, 8.1229, 8.1355, 8.1458, 8.1547]

    wax_mu_sanvito = [0.09811, 0.10238, 0.10665, 0.11091, 0.11518, 0.11945, 0.12372, 0.12798, 0.13225, 0.13652, 0.14079,
                      0.14506, 0.14932, 0.15359, 0.15786, 0.16213, 0.16639, 0.17066, 0.17493, 0.1792, 0.18347, 0.18773,
                      0.192, 0.19627, 0.20054, 0.20481, 0.20907, 0.21334, 0.21761, 0.22188, 0.22614, 0.23041, 0.23468,
                      0.23895, 0.24322, 0.24748, 0.25175, 0.25602, 0.26029, 0.26455, 0.26882, 0.27309, 0.27736, 0.28163,
                      0.28589, 0.29016, 0.29443, 0.2987, 0.30296, 0.30723, 0.3115, 0.31577, 0.32004, 0.3243, 0.32857,
                      0.33284, 0.33711, 0.34137, 0.34564, 0.34991, 0.35418, 0.35845, 0.36271, 0.36698, 0.37125, 0.37552,
                      0.37979, 0.38405, 0.38832, 0.39259, 0.39686, 0.40112, 0.40539, 0.40966, 0.41393, 0.4182, 0.42246,
                      0.42673, 0.431, 0.43527, 0.43953, 0.4438, 0.44807, 0.45234, 0.45661, 0.46087, 0.46514, 0.46941,
                      0.47368, 0.47794, 0.48221, 0.48648, 0.49075, 0.49502, 0.49928, 0.50355, 0.50782, 0.51209, 0.51635,
                      0.52062, 0.52489, 0.52916, 0.53343, 0.53769, 0.54196, 0.54623, 0.5505, 0.55477, 0.55903, 0.5633,
                      0.56757, 0.57184, 0.5761, 0.58037, 0.58464, 0.58891, 0.59318, 0.59744, 0.60171, 0.60598, 0.61025,
                      0.61451, 0.61878, 0.62305, 0.62732, 0.63159, 0.63585, 0.64012, 0.64439, 0.64866, 0.65292, 0.65719,
                      0.66146, 0.66573, 0.67, 0.67426, 0.67853, 0.6828, 0.68707, 0.69133, 0.6956, 0.69987, 0.70414,
                      0.70841, 0.71267, 0.71694, 0.72121, 0.72548, 0.72975, 0.73401, 0.73828, 0.74255, 0.74682, 0.75108,
                      0.75535, 0.75962, 0.76389, 0.76816, 0.77242, 0.77669, 0.78096, 0.78523, 0.78949, 0.79376, 0.79803,
                      0.8023, 0.80657, 0.81083, 0.8151, 0.81937, 0.82364, 0.8279, 0.83217, 0.83644, 0.84071, 0.84498,
                      0.84924, 0.85351, 0.85778, 0.86205, 0.86631, 0.87058, 0.87485, 0.87912, 0.88339, 0.88765, 0.89192,
                      0.89619, 0.90046, 0.90473, 0.90899, 0.91326, 0.91753, 0.9218, 0.92606, 0.93033, 0.9346, 0.93887,
                      0.94314, 0.9474, 0.95167, 0.95594, 0.96021, 0.96447, 0.96874, 0.97301, 0.97728, 0.98155, 0.98581,
                      0.99008]
    wax_t0_sanvito = [3.77878, 3.77803, 3.77688, 3.7749, 3.77169, 3.76679, 3.75974, 3.75002, 3.73726, 3.72165, 3.70346,
                      3.68288, 3.66007, 3.63509, 3.60771, 3.57701, 3.54332, 3.50814, 3.47255, 3.43683, 3.40104, 3.36519,
                      3.32931, 3.29341, 3.2575, 3.2216, 3.18574, 3.14996, 3.11432, 3.07892, 3.0439, 3.01057, 2.98111,
                      2.95624, 2.93542, 2.91777, 2.90249, 2.88879, 2.87613, 2.86418, 2.85275, 2.84167, 2.8308, 2.82003,
                      2.8092, 2.79815, 2.78668, 2.77484, 2.7628, 2.75089, 2.73961, 2.72963, 2.72157, 2.71573, 2.71173,
                      2.70901, 2.70709, 2.70568, 2.70457, 2.70368, 2.70291, 2.70223, 2.70161, 2.70103, 2.7005, 2.7,
                      2.69953, 2.69908, 2.69866, 2.69826, 2.69788, 2.69752, 2.69718, 2.69686, 2.69655, 2.69627, 2.696,
                      2.69575, 2.69552, 2.69532, 2.69513, 2.69497, 2.69483, 2.6947, 2.69457, 2.69442, 2.69424, 2.69402,
                      2.69373, 2.69337, 2.69291, 2.69235, 2.69167, 2.69086, 2.68993, 2.68887, 2.68769, 2.68638, 2.68496,
                      2.68344, 2.68183, 2.68013, 2.67836, 2.67652, 2.67461, 2.67264, 2.67062, 2.66854, 2.66641, 2.66424,
                      2.66202, 2.65975, 2.65744, 2.65509, 2.65269, 2.65026, 2.64779, 2.64527, 2.64272, 2.64013, 2.6375,
                      2.63483, 2.63213, 2.62939, 2.62662, 2.62383, 2.621, 2.61815, 2.61528, 2.61238, 2.60947, 2.60654,
                      2.6036, 2.60065, 2.59769, 2.59473, 2.59177, 2.58882, 2.58588, 2.58297, 2.58011, 2.57732, 2.57464,
                      2.57211, 2.56973, 2.56754, 2.56556, 2.56381, 2.56228, 2.56099, 2.55992, 2.55909, 2.55847, 2.55805,
                      2.55783, 2.55779, 2.55792, 2.55819, 2.55859, 2.55911, 2.55976, 2.56051, 2.56137, 2.56233, 2.5634,
                      2.56457, 2.56584, 2.56722, 2.56871, 2.57031, 2.57202, 2.57387, 2.57586, 2.578, 2.58031, 2.58283,
                      2.58559, 2.58865, 2.59204, 2.59583, 2.60008, 2.60485, 2.6102, 2.61615, 2.6227, 2.62982, 2.63743,
                      2.64543, 2.65372, 2.66222, 2.67092, 2.67981, 2.6889, 2.69823, 2.7078, 2.71764, 2.72779, 2.7383,
                      2.74922, 2.76065, 2.77269, 2.78556, 2.79954, 2.81509, 2.83266, 2.85292, 2.87669, 2.90474, 2.9366,
                      2.96913]
    wax_t1_sanvito = [4.0111, 4.01529, 4.01814, 4.01878, 4.01705, 4.01295, 4.00568, 3.99494, 3.98092, 3.96399, 3.94451,
                      3.92281, 3.89919, 3.87397, 3.84753, 3.82012, 3.79191, 3.76301, 3.73351, 3.70348, 3.67295, 3.64195,
                      3.61051, 3.57861, 3.54624, 3.51333, 3.47983, 3.44574, 3.41122, 3.37678, 3.3434, 3.31217, 3.28354,
                      3.25754, 3.23408, 3.21305, 3.19429, 3.17765, 3.16282, 3.14925, 3.1365, 3.12423, 3.11211, 3.09985,
                      3.08712, 3.07358, 3.0592, 3.04486, 3.03144, 3.01945, 3.00906, 3.00022, 2.99277, 2.98654, 2.98138,
                      2.97713, 2.97366, 2.9708, 2.96842, 2.96644, 2.96479, 2.96342, 2.9623, 2.96139, 2.96069, 2.96016,
                      2.95978, 2.95956, 2.95947, 2.9595, 2.95965, 2.95991, 2.96027, 2.96073, 2.96128, 2.96192, 2.96267,
                      2.96355, 2.96456, 2.96568, 2.96686, 2.96803, 2.96912, 2.97012, 2.97102, 2.97183, 2.97254, 2.97317,
                      2.9737, 2.97415, 2.97451, 2.97479, 2.97497, 2.97508, 2.97509, 2.97502, 2.97486, 2.97462, 2.97428,
                      2.97386, 2.97336, 2.97276, 2.97207, 2.97129, 2.97042, 2.96945, 2.9684, 2.96726, 2.96604, 2.96475,
                      2.96339, 2.96196, 2.96048, 2.95894, 2.95736, 2.95572, 2.95404, 2.95232, 2.95056, 2.94877, 2.94694,
                      2.94508, 2.94319, 2.94128, 2.93934, 2.93738, 2.93541, 2.93342, 2.93141, 2.9294, 2.92739, 2.92538,
                      2.92337, 2.92137, 2.9194, 2.91745, 2.91554, 2.91368, 2.91189, 2.91017, 2.90855, 2.90705, 2.90569,
                      2.90448, 2.90344, 2.90259, 2.90194, 2.90149, 2.90124, 2.90119, 2.90132, 2.90161, 2.90207, 2.90268,
                      2.90344, 2.90434, 2.90538, 2.90655, 2.90785, 2.90928, 2.91085, 2.91255, 2.91438, 2.91635, 2.91847,
                      2.92073, 2.92315, 2.92575, 2.92853, 2.93152, 2.93472, 2.93814, 2.94179, 2.94568, 2.94982, 2.9542,
                      2.95884, 2.96373, 2.96886, 2.97424, 2.97985, 2.98571, 2.9918, 2.99813, 3.00472, 3.01156, 3.01868,
                      3.02608, 3.03379, 3.04184, 3.05026, 3.05909, 3.06838, 3.07822, 3.0887, 3.0999, 3.11185, 3.1246,
                      3.13819, 3.15265, 3.16802, 3.18427, 3.20136, 3.21912, 3.23758, 3.25728, 3.27935, 3.30638, 3.34502,
                      3.40364]
    wax_t2_sanvito = [4.23647, 4.24302, 4.24824, 4.25077, 4.24982, 4.24513, 4.23679, 4.22545, 4.212, 4.19705, 4.18102,
                      4.16425, 4.14707, 4.12983, 4.11244, 4.09388, 4.07322, 4.05007, 4.02455, 3.99702, 3.96784, 3.93738,
                      3.90601, 3.87394, 3.84131, 3.8082, 3.77465, 3.74069, 3.70632, 3.67149, 3.63611, 3.59989, 3.56231,
                      3.52649, 3.50306, 3.48402, 3.46556, 3.44734, 3.42928, 3.41136, 3.39355, 3.37588, 3.35836, 3.34106,
                      3.3241, 3.3078, 3.29262, 3.27878, 3.26541, 3.25252, 3.24085, 3.23069, 3.22216, 3.21531, 3.21016,
                      3.20657, 3.20418, 3.20271, 3.2019, 3.20159, 3.20163, 3.20188, 3.20224, 3.20261, 3.20291, 3.20313,
                      3.20331, 3.20349, 3.20371, 3.20401, 3.20442, 3.20496, 3.20567, 3.20657, 3.20769, 3.20905, 3.21065,
                      3.21244, 3.21436, 3.21634, 3.21834, 3.2203, 3.22222, 3.22405, 3.2258, 3.22745, 3.22901, 3.23046,
                      3.23182, 3.23308, 3.23424, 3.23531, 3.2363, 3.23721, 3.23805, 3.23883, 3.23955, 3.24023, 3.24087,
                      3.24146, 3.24203, 3.24256, 3.24306, 3.24353, 3.24399, 3.24442, 3.24482, 3.24521, 3.24559, 3.24594,
                      3.24628, 3.24661, 3.24692, 3.24722, 3.24751, 3.24779, 3.24807, 3.24833, 3.24858, 3.24883, 3.24908,
                      3.24932, 3.24955, 3.24978, 3.25001, 3.25024, 3.25047, 3.25069, 3.25092, 3.25115, 3.25138, 3.25161,
                      3.25184, 3.25208, 3.25232, 3.25258, 3.25284, 3.25311, 3.25339, 3.25368, 3.254, 3.25433, 3.25469,
                      3.25508, 3.25551, 3.25598, 3.2565, 3.2571, 3.25779, 3.25861, 3.2596, 3.26083, 3.26235, 3.26424,
                      3.26654, 3.26924, 3.2723, 3.27561, 3.27908, 3.28264, 3.28627, 3.28998, 3.29377, 3.29765, 3.30162,
                      3.30568, 3.30985, 3.31412, 3.31851, 3.32302, 3.32767, 3.33246, 3.33741, 3.34252, 3.34783, 3.35335,
                      3.35911, 3.36512, 3.37143, 3.37805, 3.38501, 3.39236, 3.4001, 3.40829, 3.41694, 3.42609, 3.43577,
                      3.446, 3.45682, 3.46826, 3.48034, 3.49305, 3.50636, 3.52024, 3.53467, 3.54962, 3.56507, 3.58101,
                      3.59743, 3.61433, 3.63172, 3.64961, 3.66803, 3.68704, 3.70678, 3.72745, 3.74935, 3.77272, 3.79753,
                      3.82289]
    wax_t3_sanvito = [4.00687, 4.0066, 4.00624, 4.0057, 4.0048, 4.00326, 4.00029, 3.99304, 3.97846, 3.96001, 3.94062,
                      3.92061, 3.89994, 3.87851, 3.85619, 3.83277, 3.80794, 3.78139, 3.75294, 3.72258, 3.69046, 3.65682,
                      3.62196, 3.58617, 3.54971, 3.51281, 3.47563, 3.43827, 3.40087, 3.36356, 3.32651, 3.29008, 3.25526,
                      3.22431, 3.19917, 3.17917, 3.16223, 3.14679, 3.13208, 3.11779, 3.10373, 3.08979, 3.07587, 3.0619,
                      3.04776, 3.0333, 3.01827, 3.00257, 2.98711, 2.97347, 2.96279, 2.95495, 2.94927, 2.94513, 2.94202,
                      2.93961, 2.93769, 2.93617, 2.93495, 2.934, 2.93326, 2.93272, 2.93235, 2.93213, 2.93207, 2.93213,
                      2.93233, 2.93266, 2.93311, 2.93371, 2.93444, 2.9353, 2.93629, 2.93738, 2.93856, 2.9398, 2.94107,
                      2.94235, 2.94362, 2.94486, 2.94607, 2.94723, 2.94835, 2.94942, 2.95045, 2.95142, 2.95234, 2.95321,
                      2.95402, 2.95478, 2.95547, 2.95609, 2.95665, 2.95714, 2.95755, 2.95788, 2.95813, 2.95829, 2.95834,
                      2.95829, 2.95814, 2.95788, 2.95752, 2.95707, 2.95652, 2.95589, 2.95517, 2.95437, 2.9535, 2.95256,
                      2.95155, 2.95049, 2.94937, 2.94821, 2.94701, 2.94578, 2.94452, 2.94324, 2.94195, 2.94065, 2.93935,
                      2.93805, 2.93676, 2.93548, 2.93422, 2.93299, 2.93178, 2.93061, 2.92947, 2.92838, 2.92732, 2.92632,
                      2.92536, 2.92445, 2.9236, 2.9228, 2.92206, 2.92137, 2.92075, 2.92018, 2.91969, 2.91927, 2.91893,
                      2.91866, 2.91848, 2.91838, 2.91838, 2.91847, 2.91865, 2.91895, 2.91935, 2.91986, 2.92049, 2.92124,
                      2.92212, 2.92314, 2.92429, 2.92559, 2.92705, 2.92866, 2.93045, 2.9324, 2.93452, 2.93682, 2.93929,
                      2.94194, 2.94477, 2.94779, 2.95099, 2.95438, 2.95797, 2.96177, 2.96578, 2.97001, 2.97447, 2.97918,
                      2.98413, 2.98936, 2.99487, 3.00069, 3.00684, 3.01335, 3.02024, 3.02751, 3.03519, 3.04328, 3.05181,
                      3.0608, 3.07024, 3.08016, 3.09056, 3.10146, 3.11286, 3.12475, 3.13712, 3.14996, 3.1633, 3.17715,
                      3.19159, 3.20666, 3.22243, 3.23901, 3.25652, 3.27512, 3.29507, 3.31658, 3.33979, 3.36466, 3.39079,
                      3.41737]

    wax_label = '(u_infty - dq_blade + u_induced) . nhat [m/s]'
    alpha_label = 'eff. angle of attack [deg]'
    from matplotlib.lines import Line2D
    u_total_dir_dict = {wax_label: keller_u_eff_nhat_history, alpha_label: keller_alpha_deg_history}
    u_app_dir_dict = {wax_label: u_app_nhat_history, alpha_label: alpha_app_deg_history}
    time_plot_style = {0.0:'tab:blue', 0.25: 'tab:orange', 0.50:'limegreen', 0.75:'tomato'}
    comparison_plot_style = {0.0: 'darkblue', 0.25: 'brown', 0.50: 'darkgreen', 0.75: 'darkred'}
    u_total_comparison_dict = {wax_label:{'mu': wax_mu_sanvito, 0.0:wax_t0_sanvito, 0.25: wax_t1_sanvito, 0.50:wax_t2_sanvito, 0.75:wax_t3_sanvito},
                               alpha_label: {'mu': alpha_mu_sanvito, 0.0: alpha_t0_sanvito, 0.25: alpha_t1_sanvito, 0.50: alpha_t2_sanvito, 0.75: alpha_t3_sanvito}}
    for label, u_total_history in u_total_dir_dict.items():
        fig_found, ax_found = plt.subplots(2, 2)
        ax_found = ax_found.flatten()
        fig_error, ax_error = plt.subplots(2, 2)
        ax_error = ax_error.flatten()

        tdx = 0
        for time_plot, time_color in comparison_plot_style.items():
            ax_found[tdx].plot(u_total_comparison_dict[label]['mu'], u_total_comparison_dict[label][time_plot], linestyle='-', color=time_color)
            idx = next((k for k, v in keller_time_history[0.].items() if np.isclose(float(v), time_plot + float(n_periods) - 1., atol=1e-4, rtol=0.)), None)
            # idx = next((k for k, v in keller_time_history[0.].items() if v == (time_plot + float(n_periods) - 1.)), None)
            ax_found[tdx].plot(mu_eval.squeeze(), u_app_dir_dict[label][idx], linestyle='-', color='k')
            tdx += 1

        for keller_constant, keller_linestyle in keller_plot_style.items():
            tdx = 0
            for time_plot, time_color in time_plot_style.items():
                idx = next((k for k, v in keller_time_history[keller_constant].items() if
                            np.isclose(float(v), time_plot + float(n_periods) - 1., atol=1e-4, rtol=0.)), None)
                ax_found[tdx].plot(mu_eval.squeeze(), u_total_history[keller_constant][idx].squeeze(),
                         linestyle=keller_linestyle, color=time_color)
                tdx += 1

        for time_plot, time_color in time_plot_style.items():
            tdx = 0
            for keller_constant, keller_linestyle in keller_plot_style.items():
                idx = next((k for k, v in keller_time_history[keller_constant].items() if
                            np.isclose(float(v), time_plot + float(n_periods) - 1., atol=1e-4, rtol=0.)), None)
                u_total_local_interpreted = vect_op.spline_interpolation(mu_eval.squeeze(), u_total_history[keller_constant][idx].squeeze(), u_total_comparison_dict[label]['mu'])
                error_local_interpreted = [(u_total_local_interpreted[idx] - u_total_comparison_dict[label][time_plot][idx])/u_total_comparison_dict[label][time_plot][idx] for idx in range(len(u_total_comparison_dict[label]['mu']))]
                ax_error[tdx].plot(u_total_comparison_dict[label]['mu'], error_local_interpreted,
                              linestyle=keller_linestyle, color=time_color)
                tdx += 1

        for idx in range(4):
            ax_found[idx].set_xlim([0, 1])
            if '[deg]' in label:
                ax_found[idx].set_ylim([0, 20])
            ax_found[idx].grid(True)
            ax_found[idx].set_xlabel('r/R [-]')

            ax_error[idx].set_xlim([0, 1])
            ax_error[idx].grid(True)
            ax_error[idx].set_xlabel('r/R [-]')
        fig_found.suptitle(label)
        fig_error.suptitle(label[:-6] + ': error from Sanvito (2024) [-]')

        time_handles = [
            Line2D([0], [0], color=color, lw=2, label=f'{t - float(n_periods - 1):g}')
            for t, color in time_plot_style.items()
        ]
        keller_handles = [
            Line2D([0], [0], color='black', linestyle=ls, lw=2, label=f'{k:g}')
            for k, ls in keller_plot_style.items()
        ]
        sanvito_handles = [
            Line2D([0], [0], color=color, linestyle='-', lw=2, label=f'{t - float(n_periods - 1):g}')
            for t, color in comparison_plot_style.items()
        ]
        zero_induction_handle = Line2D([0], [0], color='black', linestyle='-', lw=2,
            label='without induction'
        )
        # Row headers
        header_handles = [
            Line2D([], [], linestyle='none', label='Time [s]'),
            Line2D([], [], linestyle='none', label='Keller constant [-]'),
            Line2D([], [], linestyle='none', label='Sanvito (2024)'),
        ]
        # Matplotlib fills legend columns first
        empty = lambda: Line2D([], [], linestyle='none', label='')
        zero_label = Line2D([], [], linestyle='none', label='Without induction')
        zero_line = Line2D([0], [0], color='black', lw=2, label='')
        handles_found = header_handles.copy()
        handles_found.append(zero_label)
        for i, (th, kh, sh) in enumerate(zip(time_handles, keller_handles, sanvito_handles)):
            handles_found.extend([
                th,
                kh,
                sh,
                zero_line if i == 0 else empty()
            ])
        fig_found.legend(
            handles=handles_found,
            ncol=5,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
            fontsize=11,  # Increase from 8 to 11
            columnspacing=1.5,  # More space between columns
            handlelength=2.5,  # Longer legend lines
            handletextpad=0.7,  # More space between line and label
            labelspacing=0.5,  # More vertical spacing between rows
            borderaxespad=0.5,
        )
        fig_found.subplots_adjust(bottom=0.30, hspace=0.45)

        handles_error = header_handles[:2].copy()
        for th, kh in zip(time_handles, keller_handles):
            handles_error.extend([th, kh])
        fig_error.legend(
            handles=handles_error,
            ncol=5,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
            fontsize=11,  # Increase from 8 to 11
            columnspacing=1.5,  # More space between columns
            handlelength=2.5,  # Longer legend lines
            handletextpad=0.7,  # More space between line and label
            labelspacing=0.5,  # More vertical spacing between rows
            borderaxespad=0.5,
        )
        fig_error.subplots_adjust(bottom=0.30, hspace=0.45)

    fig = plt.figure()
    afourier_style = {'a0':'tab:blue', 'acos':'tab:orange', 'asin':'forestgreen'}
    for keller_constant, keller_linestyle in keller_plot_style.items():
        for aname, acolor in afourier_style.items():
            plt.plot(np.array([float(time) for time in keller_time_history[keller_constant].values()])[1:], np.array([float(local_dict[aname]) for local_dict in keller_step_history[keller_constant].values()]), label=aname,
                     linestyle=keller_linestyle, color=acolor)
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('a_0, a_cos, a_sin [-]')

    print_op.print_dict_as_table(step_history)
    plt.show()

    import pdb; pdb.set_trace()

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
    # maybe this is positive yaw (see image in Figure 9) or maybe this is negative yaw (see image in Figure 1)?
    gamma_yaw = -30. * np.pi / 180.
    # approx_a0 = 0.18
    approx_a0 = 0.5 - (1 - ct)**0.5 / 2.
    # approx_a0 = cas.cos(gamma_yaw) / 2. - 0.5 * (-ct + cas.cos(gamma_yaw)**2.)**0.5

    cmz = 0.
    cmy = 0.

    exterior_radius = 1.
    radius_bar = 0.7 * exterior_radius # Page 199 of branlard
    wingspan = (exterior_radius - radius_bar) * 2.

    area = np.pi * exterior_radius**2.
    rho = 1.225
    u_zero = 5.
    actuator_dynamic_pressure = 0.5 * rho * u_zero**2.

    thrust = ct * actuator_dynamic_pressure * area
    moment_y = cmy * actuator_dynamic_pressure * area * exterior_radius
    moment_z = cmz * actuator_dynamic_pressure * area * exterior_radius

    n_hat = cas.cos(gamma_yaw) * vect_op.xhat_dm() + cas.sin(gamma_yaw) * vect_op.yhat_dm()
    u_hat = vect_op.xhat_dm()
    dcm_act, dcm_wind = actuator_flow.construct_actuator_and_wind_reference_frames_from_n_and_u(n_hat, u_hat)
    z_rotor_hat = dcm_act[:, 2]
    misalignment_angle = vect_op.angle_between(u_hat, n_hat, z_rotor_hat)

    c_all, c_numer, c_denom = actuator_coeff.assemble_c_components_from_pieces(thrust, moment_y, moment_z, actuator_dynamic_pressure, area, radius_bar, wingspan)
    c_denom_diag_inv = cas.inv(cas.diag(c_denom))
    MM = actuator_coeff.get_MM_matrix()

    veri_test_table = {
        'name': 'zero_moment',
        'YawAngle': gamma_yaw * 180./np.pi,
        'nhat': n_hat,
        'uhat': u_hat,
        'RotorMisalignmentAngle': misalignment_angle * 180./np.pi,
        'CT': c_all[0],
        'Cmy': cmy,
        'Cmz': cmz,
        'ExteriorRadius': exterior_radius,
        'AirDensity': rho,
        'RotorApparentSpeed': u_zero
    }
    print_op.print_dict_as_table(veri_test_table)

    t_star = exterior_radius / u_zero
    t_star_num = exterior_radius
    t_star_den = u_zero

    step_history = {0: {'a0': approx_a0, 'acos': 0, 'asin': 0}}
    for step in range(20):
        corr = actuator_flow.get_corr_val_glauert(approx_a0, misalignment_angle)
        chi = actuator_flow.get_wake_angle_chi_jimenez(misalignment_angle, thrust, area, actuator_dynamic_pressure)
        # chi = actuator_flow.get_wake_angle_chi_coleman(misalignment_angle, approx_a0)
        LL_matr = actuator_coeff.get_LL_matrix_from_corr_and_chi(corr, chi)
        LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_from_corr_and_chi(corr, chi)

        a_all_forwards = cas.mtimes(c_denom_diag_inv, cas.mtimes(LL_matr, c_numer))
        _, _, term_2_factor, term_3 = assemble_pitt_peters_style_residual(cas.DM_eye(3), cas.DM.zeros((3, 1)), t_star, t_star_num, t_star_den, MM, LLinv, LLinv_num, LLinv_denom,
                                                                     c_all, c_numer, c_denom)
        a_all = -cas.mtimes(cas.inv(term_2_factor), term_3)
        a_dict = actuator_flow.unstack_a_all(a_all)

        step_history[step] = a_dict
        approx_a0 = a_dict['a0']

    forwards_diff = a_all_forwards - a_all
    test_0 = cas.mtimes(forwards_diff.T, forwards_diff) < eps**2.
    if not test_0:
        message = 'something went unexpectedly wrong with the pitt-peters residual assembly'
        print_op.log_and_raise_error(message=message)

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
            radial_vector = actuator_geom.assemble_radial_vector(radius, y_rotor_hat, z_rotor_hat, psi, for_kite_motion_or_induction_distribution='induction_distribution')
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

    # test -2: tentative intermediate value is a0
    tent_intermediate_value_is_a0 = (a_dict['acos']**2. < eps**2.) and ((a_dict['a0'] - tentative_intermediate_value) ** 2. < eps)
    if not tent_intermediate_value_is_a0:
        message = 'somehow we did not evaluate the sinusoid properly'
        print_op.log_and_raise_error(message=message)

    # test -1.5: asin term is always positive, because yhat and zhat flip directions so that gamma is always positive
    asin_term_is_positive = (a_dict['asin'] > eps)
    if not asin_term_is_positive:
        message = 'the misalignment-angle is somehow not positive'
        print_op.log_and_raise_error(message=message)

    # test_0 = tentative intermediate value is actually intermediate
    intermediate_value_is_intermediate = (smallest < tentative_intermediate_value) and (largest > tentative_intermediate_value)

    # test_1 = when downstream-position is zero we actually have intermediate value
    when_neither_upstream_nor_downstream_have_intermediate_values = True
    for idx, local_dict in dict.items():
        if local_dict['u']**2. < eps:
            when_neither_upstream_nor_downstream_have_intermediate_values = when_neither_upstream_nor_downstream_have_intermediate_values and ((local_dict['a'] - tentative_intermediate_value)**2. < eps)

    # test 2 = when upstream, we have the larger induction factor
    when_upstream_we_have_smallest_a = True
    for idx, local_dict in dict.items():
        if local_dict['u'] < (-1 * eps): # we are upstream.
            when_upstream_we_have_smallest_a = when_upstream_we_have_smallest_a and ((local_dict['a'] - smallest)**2. < eps)

    # test 3 = when y is negative, we have the smaller induction factor
    when_downstream_we_have_largest_a = True
    for idx, local_dict in dict.items():
        if local_dict['u'] > eps: # we are downstream.
            when_downstream_we_have_largest_a = when_downstream_we_have_largest_a and ((local_dict['a'] - largest)**2. < eps)

    # test 4 = values are roughly similar to those printed in Hur et al
    expected = cas.DM([0.09, 0.18, 0.26])
    found = cas.DM([smallest, tentative_intermediate_value, largest])
    diff = expected - found
    values_are_roughly_similar = (cas.mtimes(diff.T, diff)) < (eps * vect_op.norm(expected))

    test = intermediate_value_is_intermediate and when_neither_upstream_nor_downstream_have_intermediate_values and when_upstream_we_have_smallest_a and when_downstream_we_have_largest_a and values_are_roughly_similar
    if not test:
        print_op.print_dict_as_table(dict)
        import pdb; pdb.set_trace()
        message = 'something went wrong with the sign conventions on the yawed/skewed-wake model.'
        print_op.log_and_raise_error(message)

    # print_op.print_dict_as_table(step_history)
    plot_induction_contour(a_dict, u_zero, exterior_radius, y_rotor_hat, z_rotor_hat, 'zero_moment')

    return None


def get_steady_asym_pitt_peters_residual(model_options, atmos, wind, system_variables, parameters, outputs, parent, architecture, label, scaling):
    variables_si = system_variables['SI']

    a_all = actuator_flow.get_a_all_var(variables_si, parent, label)
    a_all_ref = actuator_flow.get_a_all_ref(model_options, parent, scaling)
    da_dt_all = cas.DM.zeros((3, 1))
    da_dt_all_ref = cas.DM.zeros((3, 1))

    c_all, c_numer, c_denom, c_all_ref, c_num_ref, c_den_ref = actuator_coeff.get_c_all_components(model_options, atmos, wind, system_variables, parameters, outputs, parent, architecture, scaling)

    LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_val(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, scaling, label)
    MM = actuator_coeff.get_MM_matrix()
    t_star, t_star_num, t_star_den = actuator_coeff.get_t_star(variables_si, parameters, parent)

    resi_unscaled, _, _, _ = assemble_pitt_peters_style_residual(a_all, da_dt_all, t_star, t_star_num, t_star_den, MM, LLinv, LLinv_num, LLinv_denom, c_all, c_numer, c_denom)

    LLinv_ref, LLinv_num_ref, LLinv_denom_ref = actuator_coeff.get_LL_inv_matrix_ref(model_options, parent, scaling)
    t_star_ref, t_star_num_ref, t_star_den_ref = actuator_coeff.get_t_star_ref(model_options, parameters, parent,
                                                                               scaling)
    _, _, term_2_ref, term_3_ref = assemble_pitt_peters_style_residual(a_all_ref, da_dt_all_ref, t_star_ref, t_star_num_ref, t_star_den_ref, MM, LLinv_ref, LLinv_num_ref, LLinv_denom_ref,
                                                                 c_all_ref, c_num_ref, c_den_ref)

    resi_reference = term_2_ref
    term_ref_name = 'term_2'

    # resi_reference = term_3_ref
    # term_ref_name = 'term_3'

    resi = scale_resi(resi_unscaled, resi_reference, term_ref_name)
    return resi

def get_actuator_orientation_cstr(model_options, wind, parent, system_variables, parameters, architecture, scaling):
    variables_si = system_variables['SI']

    # --------------------
    # 2 * dcms (18 vars) + 3 * lengths (3 vars)
    # = 21 variables total
    # --------------------
    # 21 constraints total

    cstr_list = cstr_op.ConstraintList()

    # 9
    uhat_cstr = actuator_flow.get_wind_dcm_u_along_uzero_cstr(model_options, wind, parent, system_variables, architecture, scaling)
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

    print_op.warn_about_temporary_functionality_alteration()
    for parent in architecture.layer_nodes:
        uhat = actuator_system.get_actuator_vector_unit_var(variables_si, 'u', parent)
        sign_uhat_dot_xhat_expr = -cas.mtimes(uhat.T, vect_op.xhat_dm())
        name = 'actuator_sign_uhat_dot_xhat_' + str(parent)
        sign_uhat_dot_xhat_cstr = cstr_op.Constraint(expr=sign_uhat_dot_xhat_expr,
                                  name=name,
                                  cstr_type='ineq')
        cstr_list.append(sign_uhat_dot_xhat_cstr)

    return cstr_list



def collect_actuator_outputs(model_options, atmos, wind, system_variables, outputs, parameters, architecture, scaling):
    variables_si = system_variables['SI']
    outputs = collect_actuator_support_outputs(model_options, atmos, wind, system_variables, outputs, parameters, architecture, scaling)
    outputs = collect_actuator_induction_factor_outputs(model_options, variables_si, outputs, architecture)
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

            acos_var = cas.DM(0.)
            asin_var = cas.DM(0.)
            if 'asym' in label:
                acos_var = actuator_flow.get_acos_var(variables, parent, label)
                asin_var = actuator_flow.get_asin_var(variables, parent, label)
            outputs['actuator']['acos_' + label + str(parent)] = acos_var
            outputs['actuator']['asin_' + label + str(parent)] = asin_var

        current_label = actuator_flow.get_label(model_options)
        outputs['actuator']['a0_' + str(parent)] = outputs['actuator']['a0_' + current_label + str(parent)]
        outputs['actuator']['acos_' + str(parent)] = outputs['actuator']['acos_' + current_label + str(parent)]
        outputs['actuator']['asin_' + str(parent)] = outputs['actuator']['asin_' + current_label + str(parent)]


    return outputs


def collect_actuator_support_outputs(model_options, atmos, wind, system_variables, outputs, parameters, architecture, scaling):

    variables_si = system_variables['SI']
    kite_nodes = architecture.kite_nodes

    if 'actuator' not in list(outputs.keys()):
        outputs['actuator'] = {}

    for kite in kite_nodes:
        outputs['actuator']['rhat' + str(kite)] = actuator_geom.get_kite_radial_vector(kite, variables_si, architecture)
        outputs['actuator']['radius' + str(kite)] = actuator_geom.get_kite_radius(kite, variables_si, architecture, parameters)

    layer_parents = architecture.layer_nodes
    for parent in layer_parents:

        center = actuator_geom.get_actuator_position_var(variables_si, parent)
        velocity = geom.get_center_velocity(model_options, parent, system_variables, architecture, scaling)
        area = actuator_geom.get_area_var(variables_si, parent)
        avg_radius = actuator_geom.get_average_radius(variables_si, parent, architecture, parameters)
        nhat = actuator_system.get_actuator_vector_unit_var(variables_si, 'n', parent)
        uhat = actuator_system.get_actuator_vector_unit_var(variables_si, 'u', parent)
        yaw_angle = actuator_flow.get_gamma_var(variables_si, parent)
        q_app = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, system_variables, parent, architecture, scaling)

        outputs['actuator']['center' + str(parent)] = center
        outputs['actuator']['velocity' + str(parent)] = velocity
        outputs['actuator']['avg_angular_speed' + str(parent)] = actuator_geom.get_actuator_angular_speed(parent, variables_si, parameters, architecture)
        outputs['actuator']['area' + str(parent)] = area
        outputs['actuator']['avg_radius' + str(parent)] = avg_radius
        outputs['actuator']['nhat' + str(parent)] = nhat
        outputs['actuator']['bar_varrho' + str(parent)] = actuator_geom.get_bar_varrho_var(variables_si, parent)
        outputs['actuator']['yaw' + str(parent)] = yaw_angle
        outputs['actuator']['yaw_deg' + str(parent)] = yaw_angle * 180. / np.pi
        outputs['actuator']['dyn_pressure' + str(parent)] = q_app

        thrust = actuator_force.get_actuator_thrust_var(variables_si, parent)
        ct = actuator_coeff.get_ct_val(model_options, atmos, wind, system_variables, parent, architecture, scaling)
        outputs['actuator']['thrust' + str(parent)] = thrust
        outputs['actuator']['ct' + str(parent)] = ct

        outputs['actuator']['sign_nhat_dot_xhat' + str(parent)] = vect_op.sign(cas.mtimes(nhat.T, vect_op.xhat_dm()), eps=1e-8)
        outputs['actuator']['sign_uzero_dot_xhat' + str(parent)] = vect_op.sign(cas.mtimes(uhat.T, vect_op.xhat_dm()), eps=1e-8)

        outputs['actuator']['w_z_check' + str(parent)] = actuator_flow.get_wzero_parallel_z_rotor_check(variables_si, parent)
        outputs['actuator']['gamma_check' + str(parent)] = actuator_flow.get_gamma_check(variables_si, parent)
        outputs['actuator']['gamma_comp' + str(parent)] = actuator_flow.get_gamma_val(variables_si, parent)

    return outputs


def draw_actuator(ax, side, plot_dict, cosmetics, index):
    if 'actuator' in plot_dict['interpolation_si']['outputs'].keys():
        actuator_geom.draw_actuator_geometry(ax, side, plot_dict, cosmetics, index)
        actuator_flow.draw_actuator_flow(ax, side, plot_dict, cosmetics, index)
    return None

def independent_test():
    actuator_flow.test_gamma_angle_for_dcms()
    actuator_coeff.test_LL_inversion_correctly_implemented()
    test_momentum_theory_in_helicopter_conditions()
    test_signs_of_uaxi()
    test_signs_of_qasym_with_zero_moment()
    test_helicopter_values_for_qasym()
    test_signs_of_qasym_with_nonzero_moment_from_Ma()
    test_signs_of_qasym_with_nonzero_moment_from_Sun()
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
    # independent_test()

    pitch_test_oc6_35()

