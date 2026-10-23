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


def get_momentum_theory_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label, scaling):

    a_all = actuator_flow.get_a_all_var(variables, parent, label)
    a_all_ref = actuator_flow.get_a_all_ref(model_options, parent, scaling)

    c_all, c_numer, c_denom, c_all_ref, c_num_ref, c_den_ref = actuator_coeff.get_c_all_components(model_options, atmos, wind, variables, parameters,
                                                              outputs, parent, architecture, scaling)
    LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    t_star, t_star_num, t_star_den = actuator_coeff.get_t_star(variables, parameters, parent)

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


def get_unsteady_axi_pitt_peters_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label, scaling):

    a_all = actuator_flow.get_a_all_var(variables, parent, label)
    da_dt_all = actuator_flow.get_da_all_var(variables, parent, label)
    a_all_ref = actuator_flow.get_a_all_ref(model_options, parent, scaling)
    da_dt_all_ref = actuator_flow.get_da_all_ref(model_options, parent, scaling)

    c_all, c_numer, c_denom, c_all_ref, c_num_ref, c_den_ref = actuator_coeff.get_c_all_components(model_options, atmos, wind, variables, parameters,
                                                              outputs, parent, architecture, scaling)

    LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    MM = actuator_coeff.get_MM_matrix()
    t_star, t_star_num, t_star_den = actuator_coeff.get_t_star(variables, parameters, parent)

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

    t_star = 1.
    t_star_num = 1.
    t_star_den = 1.

    rise_time = 0.1

    time_start = -20.
    time_array = np.linspace(time_start, 100., 10000)

    CT_start = 0.6
    a_start = (1. - np.sqrt(1. - CT_start)) / 2.
    time_switch = 20 + time_start

    CT_switch = 0.8
    a_switch = (1. - np.sqrt(1. - CT_switch)) / 2.
    time_return_to_start = 70. + time_start

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
        LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_from_corr_and_chi(corr, chi)
        MM = actuator_coeff.get_MM_matrix()

        a_all = cas.DM(a_current)
        c_all = cas.DM(CT_current)
        c_numer = cas.DM(CT_current)
        c_denom = cas.DM(1.)
        _, term_1_factor, term_2, term_3 = assemble_pitt_peters_style_residual(a_all, cas.DM(1.), t_star, t_star_num, t_star_den, MM[0, 0], LLinv[0, 0],
                                                                     LLinv_num[0, 0], LLinv_denom, c_all[0], c_numer[0],
                                                                     c_denom[0])
        da_dt_current = -1. * (term_2 + term_3) / term_1_factor

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

    def deltaV(a_local):
        return (a_start - a_local) / (a_switch - a_start)
    deltaV_array = [deltaV(a_local) for a_local in a_array]
    tau_array = np.array(cas.DM(time_array))
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

    subset_tau_array = []
    subset_deltaV_array = []
    for tdx in range(tau_array.shape[0]):
        if tau_array[tdx] >= time_switch:
            subset_tau_array += [tau_array[tdx]]
            subset_deltaV_array += [deltaV_array[tdx]]

    import os
    import matplotlib
    if "MPLBACKEND" not in os.environ:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    color1 = 'tab:red'
    ax1.set_ylabel('thrust coefficient [-]', color=color1)
    ax1.plot(tau_array, CT_array, label='CT', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('axial induction factor [-]', color=color2)
    ax2.plot(tau_array, a_array, label='a', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig = plt.figure()
    plt.plot(x_yu_array, qaxi_yu_array, label='qaxi_yu')
    plt.plot(x_yu_array, uaxi_yu_array, label='uaxi_yu')
    plt.plot(x_yu_array, oye_yu_array, label='oye_yu')
    plt.plot(x_yu_array, ecn_yu_array, label='ecn_yu')
    plt.plot(x_yu_array, vtn_yu_array, label='vtn_yu')
    plt.plot(x_yu_array, fvwr_yu_array, label='fwrr_yu')
    plt.plot(x_yu_array, rans_yu_array, label='rans_yu')
    # notice that deltaV in figure 6 seems to have the sign switched.
    plt.plot(subset_tau_array, subset_deltaV_array, label='uaxi_awebox')
    plt.legend()
    plt.show()

    return None

def get_unsteady_asym_pitt_peters_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture, label, scaling):

    a_all = actuator_flow.get_a_all_var(variables, parent, label)
    da_dt_all = actuator_flow.get_da_all_var(variables, parent, label)
    a_all_ref = actuator_flow.get_a_all_ref(model_options, parent, scaling)
    da_dt_all_ref = actuator_flow.get_da_all_ref(model_options, parent, scaling)

    c_all, c_numer, c_denom, c_all_ref, c_num_ref, c_den_ref = actuator_coeff.get_c_all_components(model_options, atmos, wind, variables, parameters,
                                                              outputs, parent, architecture, scaling)

    LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    MM = actuator_coeff.get_MM_matrix()
    t_star, t_star_num, t_star_den = actuator_coeff.get_t_star(variables, parameters, parent)

    resi_unscaled, _, _, _ = assemble_pitt_peters_style_residual(a_all, da_dt_all, t_star, t_star_num, t_star_den, MM, LLinv, LLinv_num, LLinv_denom, c_all, c_numer, c_denom)

    LLinv_ref, LLinv_num_ref, LLinv_denom_ref = actuator_coeff.get_LL_inv_matrix_ref(model_options, parent, scaling)
    t_star_ref, t_star_num_ref, t_star_den_ref = actuator_coeff.get_t_star_ref(model_options, parameters, parent, scaling)

    _, term_1_ref, term_2_ref, term_3_ref = assemble_pitt_peters_style_residual(a_all_ref, da_dt_all_ref, t_star_ref, t_star_num_ref, t_star_den_ref, LLinv_ref, LLinv_num_ref, LLinv_denom_ref, c_all_ref, c_num_ref, c_den_ref)

    # resi_reference = term_1_ref
    # term_ref_name = 'term_1'

    resi_reference = term_2_ref
    term_ref_name = 'term_2'

    # resi_reference = term_3_ref
    # term_ref_name = 'term_3'

    resi = scale_resi(resi_unscaled, resi_reference, term_ref_name)
    return resi


def test_signs_of_qasym_with_nonzero_moment(eps=1e-4):

    # Ma L, Li Y, Zhou L, Yang D, Shen X and Du Z (2025)
    # Study on the aeroelastic performance of 15 MW wind turbine under yaw condition. Front. Energy Res. 13:1571567. doi: 10.3389/fenrg.2025.1571567
    # https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2025.1571567/full
    #

    u_zero = 10.59 # m/s, page 03
    diameter = 240. #m, page 03
    gamma_yaw = 20. * np.pi / 180. #pg 05
    hub_radius = 0.2 * (diameter/2.) # fig. 8, pg. 8

    area = np.pi * (diameter / 2.)**2.
    rho = 1.225 #guessed
    actuator_dynamic_pressure = 0.5 * rho * u_zero**2.
    exterior_radius = diameter/2.
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

    moment_y = tilt_moment
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

    step_history = {0: {'a0': approx_a0, 'acos': 0, 'asin': 0}}
    for step in range(20):
        corr = actuator_flow.get_corr_val_glauert(approx_a0, vect_op.abs(gamma_yaw))
        chi = actuator_flow.get_wake_angle_chi_jimenez(vect_op.abs(gamma_yaw), thrust, area, actuator_dynamic_pressure)
        LL_matr = actuator_coeff.get_LL_matrix_from_corr_and_chi(corr, chi)

        a_all = cas.mtimes(LL_matr, c_all)
        a_dict = actuator_flow.unstack_a_all(a_all)
        step_history[step] = a_dict

        approx_a0 = a_dict['a0']

    # print_op.print_dict_as_table(step_history)
    # for positive gamma:
    # [ a0   ]      [ + & 0 & - ] [ + ]   [ + ]
    # [ acos ] =    [ 0 & - & 0 ] [ + ] = [ - ]
    # [ asin ]      [ + & 0 & - ] [ - ] = [ + ]
    white_line_in_same_quadrant_as_paper_plot = (a_dict['a0'] > 0) and (a_dict['acos'] < 0) and (a_dict['asin'] > 0)
    if not white_line_in_same_quadrant_as_paper_plot:
        message = 'tilt and yaw moment not turning the induction contour in expected direction'
        print_op.log_and_raise_error(message)

    # plot_induction_contour(a_dict, u_zero, exterior_radius, y_rotor_hat, z_rotor_hat, 'nonzero_moment')

    return None

def plot_induction_contour(a_dict, u_zero, exterior_radius, y_rotor_hat, z_rotor_hat, name_string):
    dict = {}
    idx = 0
    for mu in np.linspace(0., 1., 30):
        for psi in np.linspace(0., 2. * np.pi, 361):

            radius = mu * exterior_radius
            radial_vector = actuator_geom.assemble_radial_vector(radius, y_rotor_hat, z_rotor_hat, psi, for_kite_motion_or_induction_distribution='induction_distribution')
            a_local = actuator_flow.assemble_asym_local_induction_factor(a_dict, psi, mu)
            ui = u_zero * (-1.) * a_local

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

    cmap = 'RdYlBu_r'

    fig, ax = plt.subplots()
    contour = ax.tricontourf(y_array, z_array, a_array, levels=100, cmap=cmap) #, vmin=-6, vmax=-2.5)
    plt.tricontour(y_array, z_array, a_array, levels=100, cmap=cmap)
    # plt.clim(-6, -2.5)
    fig.colorbar(contour, ax=ax, label='Normal-direction induction factor [-]')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_aspect('equal')
    plt.savefig(base_name + "_a.pdf", format="pdf")

    fig, ax = plt.subplots(
        subplot_kw={'projection': 'polar'}
    )
    ax.set_theta_zero_location('N')     # Set 0° at the top (North)
    ax.set_theta_direction(-1)     # Set angles to increase clockwise
    ax.set_thetagrids(np.arange(0, 360, 30))
    contour = ax.tricontourf(psi_array, mu_array, ui_array, levels=100, cmap=cmap) #, vmin=-6, vmax=-2.5)
    plt.tricontour(psi_array, mu_array, ui_array, levels=100, cmap=cmap)
    # add the epigraphs to show rotation of the induction field
    ax.tricontour(psi_array, mu_array, ui_array, levels=[u_zero * (-1.) * a_dict['a0']], colors='black', linewidths=1.5, linestyles='solid', zorder=10)
    # Dashed reference line at psi = 0
    ax.plot([0, 0], [0, 1], color='black', linestyle='--', linewidth=1.5, zorder=10)
    # plt.clim(-6, -2.5)
    fig.colorbar(contour, ax=ax, label='Normal induced velocity [m/s]')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_aspect('equal')
    plt.savefig(base_name + "_ui.pdf", format="pdf")

    plt.show()
    return None


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

    c_all, c_numer, c_denom = actuator_coeff.assemble_c_components_from_pieces(thrust, moment_y, moment_z, actuator_dynamic_pressure, area, radius_bar, wingspan)
    c_denom_diag_inv = cas.inv(cas.diag(c_denom))
    MM = actuator_coeff.get_MM_matrix()

    t_star = exterior_radius / u_zero
    t_star_num = exterior_radius
    t_star_den = u_zero

    step_history = {0: {'a0': approx_a0, 'acos': 0, 'asin': 0}}
    for step in range(20):
        corr = actuator_flow.get_corr_val_glauert(approx_a0, vect_op.abs(gamma_yaw))
        chi = actuator_flow.get_wake_angle_chi_jimenez(vect_op.abs(gamma_yaw), thrust, area, actuator_dynamic_pressure)
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
    # plot_induction_contour(a_dict, u_zero, exterior_radius, y_rotor_hat, z_rotor_hat, 'zero_moment')

    return None


def get_steady_asym_pitt_peters_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture, label, scaling):

    a_all = actuator_flow.get_a_all_var(variables, parent, label)
    a_all_ref = actuator_flow.get_a_all_ref(model_options, parent, scaling)
    da_dt_all = cas.DM.zeros((3, 1))
    da_dt_all_ref = cas.DM.zeros((3, 1))

    c_all, c_numer, c_denom, c_all_ref, c_num_ref, c_den_ref = actuator_coeff.get_c_all_components(model_options, atmos, wind, variables, parameters, outputs, parent, architecture, scaling)

    LLinv, LLinv_num, LLinv_denom = actuator_coeff.get_LL_inv_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    MM = actuator_coeff.get_MM_matrix()
    t_star, t_star_num, t_star_den = actuator_coeff.get_t_star(variables, parameters, parent)

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

    if 'actuator' not in list(outputs.keys()):
        outputs['actuator'] = {}

    for kite in kite_nodes:
        outputs['actuator']['rhat' + str(kite)] = actuator_geom.get_kite_radial_vector(kite, variables, architecture)
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
    # test_signs_of_uaxi()
    test_signs_of_qasym_with_zero_moment()
    test_signs_of_qasym_with_nonzero_moment()
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

    test_signs_of_uaxi()

