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
"""
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-21
- edit: jochem de schutter, alu-fr 2019
"""

import casadi.tools as cas
import numpy as np

from awebox.logger.logger import Logger as awelogger

import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom
import awebox.mdl.aero.induction_dir.actuator_dir.force as actuator_force
import awebox.mdl.aero.induction_dir.actuator_dir.system as actuator_system
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools
import awebox.mdl.aero.geometry_dir.geometry as geom
import awebox.viz.tools as viz_tools

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.struct_operations as struct_op

## variables

def get_a_var_type(label):
    """ Extract variable type of average induction factor.
        steady: algebraic variable
        unsteady: differential state"""

    if label[0] == 'q':
        var_type = 'z'
    elif label[0] == 'u':
        var_type = 'x'
    else:
        message = "Invalid steadyness option (" + str(label[0]) + ") for actuator disk model chosen"
        print_op.log_and_raise_error(message)

    return var_type


def get_local_a_var(variables, kite, parent):
    prefix = actuator_system.get_actuator_var_name_prefix()
    local_a = variables['z'][prefix + 'local_a' + str(kite) + str(parent)]
    return local_a

def get_a0_var(variables_si, parent, label):
    var_type = get_a_var_type(label)
    prefix = actuator_system.get_actuator_var_name_prefix()
    var_name = prefix + 'a0_' + label + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var


def get_acos_var(variables_si, parent, label):
    var_type = get_a_var_type(label)
    prefix = actuator_system.get_actuator_var_name_prefix()
    var_name = prefix + 'acos_' + label + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var


def get_asin_var(variables_si, parent, label):
    var_type = get_a_var_type(label)
    prefix = actuator_system.get_actuator_var_name_prefix()
    var_name = prefix + 'asin_' + label + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var


def get_a_all_var(variables, parent, label):
    a_dict = {}
    a_dict['a0'] = get_a0_var(variables, parent, label)
    if 'asym' in label:
        a_dict['acos'] = get_acos_var(variables, parent, label)
        a_dict['asin'] = get_asin_var(variables, parent, label)
    a_all = restack_a_dict(a_dict)
    return a_all

def unstack_a_all(a_all):
    if a_all.shape[0] == 3:
        return {'a0': a_all[0], 'acos': a_all[1], 'asin': a_all[2]}
    else:
        return {'a0': a_all[0]}

def restack_a_dict(a_dict):
    a_all = a_dict['a0']
    if 'acos' in a_dict.keys():
        a_all = cas.vertcat(a_all, a_dict['acos'])
    if 'asin' in a_dict.keys():
        a_all = cas.vertcat(a_all, a_dict['asin'])
    return a_all

def get_a_all_ref(model_options, parent, scaling):
    label = get_label(model_options)

    var_type = get_a_var_type(label)
    prefix = actuator_system.get_actuator_var_name_prefix()
    a0_name = prefix + 'a0_' + label + str(parent)
    acos_name = prefix + 'acos_' + label + str(parent)
    asin_name = prefix + 'asin_' + label + str(parent)

    a_dict = {}
    a_dict['a0'] = scaling[var_type, a0_name, 0]
    if 'asym' in label:
        a_dict['acos'] = scaling[var_type, acos_name, 0]
        a_dict['asin'] = scaling[var_type, asin_name, 0]
    a_all_ref = restack_a_dict(a_dict)

    return a_all_ref

def get_da_all_ref(model_options, parent, scaling):
    label = get_label(model_options)

    var_type = 'xdot'
    prefix = actuator_system.get_actuator_var_name_prefix()
    a0_name = 'd' + prefix + 'a0_' + label + str(parent)
    acos_name = 'd' + prefix + 'acos_' + label + str(parent)
    asin_name = 'd' + prefix + 'asin_' + label + str(parent)

    a_dict = {}
    a_dict['a0'] = scaling[var_type, a0_name, 0]
    if 'asym' in label:
        a_dict['acos'] = scaling[var_type, acos_name, 0]
        a_dict['asin'] = scaling[var_type, asin_name, 0]
    a_all_ref = restack_a_dict(a_dict)

    return a_all_ref

def get_da_var(variables, parent, label):
    prefix = actuator_system.get_actuator_var_name_prefix()
    da_var = variables['xdot']['d' + prefix + 'a0_' + label + str(parent)]
    return da_var

def get_dacos_var(variables, parent, label):
    prefix = actuator_system.get_actuator_var_name_prefix()
    dacos_var = variables['xdot']['d' + prefix + 'acos_' + label + str(parent)]
    return dacos_var

def get_dasin_var(variables, parent, label):
    prefix = actuator_system.get_actuator_var_name_prefix()
    dasin_var = variables['xdot']['d' + prefix + 'asin_' + label + str(parent)]
    return dasin_var

def get_da_all_var(variables, parent, label):
    a_dict = {}
    a_dict['a0'] = get_da_var(variables, parent, label)
    if 'asym' in label:
        a_dict['acos'] = get_dacos_var(variables, parent, label)
        a_dict['asin'] = get_dasin_var(variables, parent, label)
    a_all = restack_a_dict(a_dict)
    return a_all

def get_wind_dcm_var(variables_si, parent):
    return actuator_system.get_actuator_dcm_var(variables_si, 'u', parent)

def get_gamma_var(variables, parent):
    var_type = 'z'
    prefix = actuator_system.get_actuator_var_name_prefix()
    var_name = prefix + 'gamma' + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables, var_type, var_name)
    return var


def get_cosgamma_var(variables, parent):
    gamma = get_gamma_var(variables, parent)
    var = cas.cos(gamma)
    return var


def get_singamma_var(variables, parent):
    gamma = get_gamma_var(variables, parent)
    var = cas.sin(gamma)
    return var



## residuals

def get_gamma_cstr(parent, variables, scaling):
    # cos = u / hypotenuse
    # sin = v / hypotenuse
    # tan = sin / cos = v / u

    uzero_hat_var = actuator_system.get_actuator_vector_unit_var(variables, 'u', parent)
    vzero_hat_var = actuator_system.get_actuator_vector_unit_var(variables, 'v', parent)

    n_hat_var = actuator_system.get_actuator_vector_unit_var(variables, 'n', parent)

    u_comp = cas.mtimes(n_hat_var.T, uzero_hat_var)
    v_comp = cas.mtimes(n_hat_var.T, vzero_hat_var)

    gamma_var = get_gamma_var(variables, parent)
    resi = u_comp * cas.sin(gamma_var) - v_comp * cas.cos(gamma_var)

    name = 'actuator_gamma_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi,
                              name=name,
                              cstr_type='eq')

    return cstr


def check_that_gamma_is_consistent(variables_si, parent, epsilon=1.e-4):
    gamma_var = get_gamma_var(variables_si, parent)
    cosgamma_var = get_cosgamma_var(variables_si, parent)
    singamma_var = get_singamma_var(variables_si, parent)

    cos_check = (cosgamma_var - cas.cos(gamma_var))**2. < epsilon**2.
    sin_check = (singamma_var - cas.sin(gamma_var))**2. < epsilon**2.

    n_hat_var = actuator_system.get_actuator_vector_unit_var(variables_si, 'n', parent)
    uzero_hat_var = actuator_system.get_actuator_vector_unit_var(variables_si, 'u', parent)
    z_hat_var = actuator_system.get_actuator_vector_unit_var(variables_si, 'z', parent)
    angle_check = (gamma_var - calculate_gamma({'n_hat': n_hat_var, 'u_hat': uzero_hat_var, 'z_hat': z_hat_var}))**2. < epsilon**2.

    if not (cos_check and sin_check and angle_check):
        message = 'something went wrong with the actuator gamma angle (between u_zero and n_hat)'
        print_op.log_and_raise_error(message)

    return None


def get_wind_dcm_ortho_cstr(parent, variables_si):

    # rotation matrix is in SO3 = 6 constraints
    wind_dcm_var = get_wind_dcm_var(variables_si, parent).reshape((3, 3))
    ortho_matr = cas.mtimes(wind_dcm_var.T, wind_dcm_var) - np.eye(3)
    f_ortho = vect_op.upper_triangular_inclusive(ortho_matr)

    name = 'actuator_wind_dcm_ortho_' + str(parent)
    cstr = cstr_op.Constraint(expr=f_ortho,
                              name=name,
                              cstr_type='eq')

    return cstr


def get_wind_dcm_u_along_uzero_cstr(model_options, wind, parent, system_variables, architecture, scaling):
    variables_si = system_variables['SI']

    u_vec_val = general_flow.get_vec_u_zero(model_options, wind, parent, system_variables, architecture, scaling)
    u_hat_var = actuator_system.get_actuator_vector_unit_var(variables_si, 'u', parent)
    u_length_var = actuator_system.get_actuator_vector_length_var(variables_si, 'u', parent)

    resi_align = u_vec_val - u_hat_var * u_length_var

    var_name = actuator_system.get_actuator_vector_length_name('u', parent)
    factor = struct_op.var_si_to_scaled('z', var_name, cas.DM(1.), scaling)
    resi_scaled = resi_align * factor

    name = 'actuator_uhat_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi_scaled,
                              name=name,
                              cstr_type='eq')

    return cstr

def get_wind_dcm_w_along_act_dcm_z_cstr(model_options, wind, parent, variables_si, architecture, scaling):

    w_hat_var = actuator_system.get_actuator_vector_unit_var(variables_si, 'w', parent)
    z_hat_var = actuator_system.get_actuator_vector_unit_var(variables_si, 'z', parent)
    z_length_var = actuator_system.get_actuator_vector_length_var(variables_si, 'z', parent)

    resi_align = w_hat_var - z_hat_var * z_length_var

    var_name = actuator_system.get_actuator_vector_length_name('z', parent)
    factor = struct_op.var_si_to_scaled('z', var_name, cas.DM(1.), scaling)
    resi_scaled = resi_align * factor

    name = 'actuator_what_parallel_zhat_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi_scaled,
                              name=name,
                              cstr_type='eq')
    return cstr


def check_that_uzero_has_positive_component_in_dominant_wind_direction(wind, variables_si, parent, epsilon=1e-5):
    u_hat_var = actuator_system.get_actuator_vector_unit_var(variables_si, 'u', parent)
    wind_dir = wind.get_wind_direction()

    if cas.mtimes(u_hat_var.T, wind_dir) < epsilon:
        message = 'something went wrong when setting the actuator u_zero. u_zero is expected to be at-least-vaguely downstream'
        print_op.log_and_raise_error(message)

    return None

def get_wzero_parallel_z_rotor_check(variables_si, parent):
    z_rotor_hat = actuator_system.get_actuator_vector_unit_var(variables_si, 'z', parent)
    w_hat_var = actuator_system.get_actuator_vector_unit_var(variables_si, 'w', parent)

    check = cas.mtimes(w_hat_var.T, z_rotor_hat) - 1.
    return check

def check_that_wzero_is_parallel_to_z_rotor(variables_si, parent, epsilon=1e-5):
    if get_wzero_parallel_z_rotor_check(variables_si, parent) ** 2. > epsilon ** 2.:
        message = 'something went wrong when setting the actuator and wind dcms. wzero is not parallel to rotor zhat'
        print_op.log_and_raise_error(message)
    return None


def estimate_reference_actuator_and_wind_reference_frames(model_options, wind, scaling):
    altitude = model_options['wind']['estimated_operating_altitude']
    u_alt_vec = wind.get_velocity(altitude)
    l_t = scaling['x', 'l_t']
    inclination = cas.asin(altitude / l_t)
    n_hat = cas.cos(inclination) * vect_op.xhat_dm() + cas.sin(inclination) * vect_op.zhat_dm()

    f_reelout_loyd = 1./3.
    u_reelout_vec = vect_op.norm(u_alt_vec) * f_reelout_loyd * n_hat
    u_zero_est = u_alt_vec - u_reelout_vec
    u_hat = vect_op.normalize(u_zero_est)

    dcm_act, dcm_wind = construct_actuator_and_wind_reference_frames_from_n_and_u(n_hat, u_hat)
    return dcm_act, dcm_wind


def construct_actuator_and_wind_reference_frames_from_n_and_u(n_hat, u_hat):
    # see:
    # Fig. 3.47 on page 97 of Burton et al, Wind Energy Handbook,
    # Fig. B1-B2, pages 131-132 of Suzuki, A. Application of dynamic inﬂow theory to wind turbine rotors; The University of Utah, 2000.

    z_hat = vect_op.normed_cross(u_hat, n_hat)
    y_hat = vect_op.normed_cross(z_hat, n_hat)

    w_hat = z_hat
    v_hat = vect_op.normed_cross(w_hat, u_hat)

    dcm_act = cas.horzcat(n_hat, y_hat, z_hat)
    dcm_wind = cas.horzcat(u_hat, v_hat, w_hat)
    return dcm_act, dcm_wind

def test_gamma_angle_for_dcms(thresh=1e-2):

    # see Fig. 3.47 on page 97 of Burton et al, Wind Energy Handbook
    u_hat = vect_op.xhat_dm()
    z_hat = vect_op.zhat_dm()

    gamma = -17. * np.pi / 180.
    n_hat = cas.cos(gamma) * vect_op.xhat_dm() + cas.sin(gamma) * vect_op.yhat_dm()
    test_1 = (calculate_gamma({'n_hat': n_hat, 'u_hat': u_hat, 'z_hat': z_hat}) - gamma)**2. < thresh**2.

    gamma = 17. * np.pi / 180.
    n_hat = cas.cos(gamma) * vect_op.xhat_dm() + cas.sin(gamma) * vect_op.yhat_dm()
    test_2 = (calculate_gamma({'n_hat': n_hat, 'u_hat': u_hat, 'z_hat': z_hat}) - gamma)**2. < thresh**2.

    w_hat = vect_op.zhat_dm()
    dcm_act, dcm_wind = construct_actuator_and_wind_reference_frames_from_n_and_u(n_hat, u_hat)
    w_diff = dcm_wind[:, 2] - w_hat
    test_3 = cas.mtimes(w_diff.T, w_diff) < thresh**2.

    if not (test_1 and test_2 and test_3):
        message = 'something went wrong with the reference frame or yaw angle calculations'
        print_op.log_and_raise_error(message)
    return None

def get_induction_factor_assignment_cstr(model_options, variables, kite, parent, scaling):
    a_var = get_local_a_var(variables, kite, parent)

    label = get_label(model_options)
    a_val = get_local_induction_factor(model_options, variables, kite, parent, label)

    resi_si = a_var - a_val

    var_type = 'z'
    prefix = actuator_system.get_actuator_var_name_prefix()
    var_name = prefix + 'local_a' + str(kite) + str(parent)
    resi_scaled = struct_op.var_si_to_scaled(var_type, var_name, resi_si, scaling)

    name = 'actuator_a_assignment_' + str(kite)
    cstr = cstr_op.Constraint(expr=resi_scaled,
                              name=name,
                              cstr_type='eq')
    return cstr


## values
def calculate_gamma(ehat_dict):
    gamma = vect_op.angle_between(ehat_dict['u_hat'], ehat_dict['n_hat'], ehat_dict['z_hat'])
    return gamma

def get_gamma_val(variables, parent):
    u_hat_var = actuator_system.get_actuator_vector_unit_var(variables, 'u', parent)
    n_hat_var = actuator_system.get_actuator_vector_unit_var(variables, 'n', parent)
    z_hat_var = actuator_system.get_actuator_vector_unit_var(variables, 'z', parent)
    gamma = calculate_gamma({'n_hat': n_hat_var, 'u_hat': u_hat_var, 'z_hat': z_hat_var})
    return gamma

def get_gamma_check(variables, parent):
    gamma_val = vect_op.abs(get_gamma_val(variables, parent))
    gamma_var = vect_op.abs(get_gamma_var(variables, parent))
    check = gamma_val - gamma_var
    norm = cas.mtimes(check.T, check)
    return norm

## references


def get_qzero_ref(atmos, wind):
    scale = 5.
    rho_ref = atmos.get_density_ref()
    uinfty_ref = wind.get_speed_ref()
    qzero_ref = .5 * rho_ref * uinfty_ref**2. * scale
    return qzero_ref

def get_uzero_vec_length_ref(parent, scaling):
    var_name = actuator_system.get_actuator_vector_length_name('u', parent)
    return scaling['z', var_name, 0]

def assemble_asym_local_induction_factor(a_dict, psi, mu):
    return a_dict['a0'] + a_dict['acos'] * cas.cos(psi) * mu + a_dict['asin'] * cas.sin(psi) * mu

def get_local_induction_factor(model_options, variables, kite, parent, label):

    if 'asym' in label:

        psi = actuator_geom.get_psi_var(variables, kite, parent)

        if model_options['aero']['actuator']['asym_radial_linearity']:

            # version given in original Kinner / Pitt&Peters / Burton et al.
            # applied to annuli in Suzuki 2000 (3.7, page 30): middle-radius-of-annulus/total-disk-exterior-radius
            varrho_var = actuator_geom.get_varrho_var(variables, kite, parent)
            bar_varrho_var = actuator_geom.get_bar_varrho_var(variables, parent)
            mu = varrho_var / (bar_varrho_var + 0.5)

        else:
            mu = 1.
            # see Suzuki2000 (phd thesis), page 27-30
            # Application of Dynamic Inflow Theory to Wind Turbine Rotors
            # for motivation for evaluating at the edges of the "annulus"
            # also: seems to lead to less optimizer-trickery.

        a_uni = get_a0_var(variables, parent, label)
        acos = get_acos_var(variables, parent, label)
        asin = get_asin_var(variables, parent, label)
        a_local = assemble_asym_local_induction_factor({'a0': a_uni, 'acos': acos, 'asin': asin}, psi, mu)

    elif 'axi' in label:
        a_local = get_a0_var(variables, parent, label)

    else:
        message = 'an unfamiliar actuator model label was entered when computing the local induction factor'
        print_op.log_and_raise_error(message)

    return a_local


def get_local_induced_velocity(model_options, system_variables, parameters, architecture, wind, kite, parent, scaling, label):
    variables_si = system_variables['SI']

    vec_u_zero_val = general_flow.get_vec_u_zero(model_options, wind, parent, system_variables, architecture, scaling)
    u_zero = vect_op.norm(vec_u_zero_val)
    n_hat_var = actuator_system.get_actuator_vector_unit_var(variables_si, 'n', parent)

    a_val = get_local_induction_factor(model_options, variables_si, kite, parent, label)
    u_ind = -1. * a_val * u_zero * n_hat_var

    return u_ind


def get_kite_induced_velocity(model_options, system_variables, parameters, architecture, wind, kite, parent, scaling):
    label = get_label(model_options)
    u_ind_kite = get_local_induced_velocity(model_options, system_variables, parameters, architecture, wind, kite, parent, scaling, label)
    return u_ind_kite

def get_kite_effective_velocity_from_a_var(model_options, system_variables, parameters, architecture, wind, kite, parent, scaling):
    variables_si = system_variables['SI']
    u_app_kite = general_flow.get_kite_apparent_velocity(variables_si, wind, kite, parent)
    u_ind_kite = get_kite_induced_velocity(model_options, system_variables, parameters, architecture, wind, kite, parent, scaling)
    u_eff_kite = u_app_kite + u_ind_kite
    return u_eff_kite

def get_actuator_dynamic_pressure(model_options, atmos, wind, system_variables, parent, architecture, scaling):
    variables_si = system_variables['SI']
    center = actuator_geom.get_actuator_position_var(variables_si, parent)
    rho_infty = atmos.get_density(center[2])

    u_vec_val = general_flow.get_vec_u_zero(model_options, wind, parent, system_variables, architecture, scaling)

    qzero = 0.5 * rho_infty * cas.mtimes(u_vec_val.T, u_vec_val)

    return qzero


def get_wake_angle_chi_equal(gamma_var):
    return gamma_var

def get_wake_angle_chi_coleman(gamma_var, a0_var):
    chi = (0.6 * a0_var + 1.) * gamma_var
    return chi

def get_wake_angle_chi_jimenez(gamma_var, thrust, area, qzero):
    cosgamma = cas.cos(gamma_var)
    singamma = cas.sin(gamma_var)
    ct_val = thrust / area / qzero
    chi = gamma_var + 0.5 * ct_val * cosgamma**2. * singamma
    return chi

def get_wake_angle_chi(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, scaling, label):
    variables_si = system_variables['SI']

    gamma_var = get_gamma_var(variables_si, parent)
    a0_var = get_a0_var(variables_si, parent, label)
    thrust = actuator_force.get_actuator_thrust_var(variables_si, parent)
    area = actuator_geom.get_actuator_area(parent, variables_si, parameters)
    qzero = get_actuator_dynamic_pressure(model_options, atmos, wind, system_variables, parent, architecture, scaling)

    wake_skew = model_options['aero']['actuator']['wake_skew']

    if wake_skew == 'equal':
        chi_val = get_wake_angle_chi_equal(gamma_var)
    elif wake_skew == 'coleman':
        chi_val = get_wake_angle_chi_coleman(gamma_var, a0_var)
    elif wake_skew == 'jimenez':
        chi_val = get_wake_angle_chi_jimenez(gamma_var, thrust, area, qzero)
    elif wake_skew == 'not_in_use':
        chi_val = 0.
    else:
        message = 'unknown wake skew angle (chi) model selected'
        print_op.log_and_raise_error(message)

    return chi_val


def get_actuator_comparison_labels(model_options):
    comparison_labels = model_options['aero']['induction']['comparison_labels']

    actuator_comp_labels = []
    for label in comparison_labels:
        if label[:3] == 'act':
            actuator_comp_labels += [label[4:]]

    return actuator_comp_labels

def get_label(model_options):
    steadyness = model_options['induction']['steadyness']
    symmetry = model_options['induction']['symmetry']

    accepted_steadyness_dict = {'quasi-steady':'q', 'steady':'q', 'unsteady':'u'}
    accepted_symmetry_dict = {'axisymmetric': 'axi', 'asymmetric': 'asym'}
    
    if (steadyness in accepted_steadyness_dict.keys()) and (symmetry in accepted_symmetry_dict.keys()):
        label = accepted_steadyness_dict[steadyness] + accepted_symmetry_dict[symmetry]

    else:
        message = 'unrecognized actuator option (' + steadyness + ', ' + symmetry + ') indicated. available ' \
                'options are: ' + repr(accepted_steadyness_dict.keys()) + ' and ' + repr(accepted_symmetry_dict.keys())
        print_op.log_and_raise_error(message)

    return label


def get_corr_val_axisym(a0_var):
    corr_val = (1. - a0_var)
    return corr_val

def get_corr_val_glauert(a0_var, gamma_var):
    cosgamma = cas.cos(gamma_var)
    corr_val = cas.sqrt( (1. - a0_var * (2. * cosgamma - a0_var)) )
    return corr_val

def get_corr_val_coleman(a0_var, gamma_var, chi):
    singamma = cas.sin(gamma_var)
    cosgamma = cas.cos(gamma_var)
    corr_val = cosgamma + cas.tan(chi / 2.) * singamma - a0_var / (cas.cos(chi / 2.)**2.)
    return corr_val

def get_corr_val_simple(a0_var, gamma_var):
    cosgamma = cas.cos(gamma_var)
    corr_val = (cosgamma - a0_var)
    return corr_val

def get_corr_val(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, scaling, label):
    variables_si = system_variables['SI']

    # options here from Chapter 4 of Wind Energy Handbook
    # Tony Burton, Nick Jenkins, David Sharpe, and Ervin Bossanyi

    a0_var = get_a0_var(variables_si, parent, label)
    gamma_var = get_gamma_var(variables_si, parent)
    chi = get_wake_angle_chi(model_options, atmos, wind, system_variables, outputs, parameters, parent, architecture, scaling, label)

    actuator_skew = model_options['aero']['actuator']['actuator_skew']
    if actuator_skew == 'not_in_use':
        corr_val = get_corr_val_axisym(a0_var)
    elif actuator_skew == 'glauert':
        corr_val = get_corr_val_glauert(a0_var, gamma_var)
    elif actuator_skew == 'coleman':
        corr_val = get_corr_val_coleman(a0_var, gamma_var, chi)
    elif actuator_skew == 'simple':
        corr_val = get_corr_val_simple(a0_var, gamma_var)
    else:
        message = 'unknown actuator angle correction model selected: ' + actuator_skew
        print_op.log_and_raise_error(message)
    return corr_val


def draw_actuator_flow(ax, side, plot_dict, cosmetics, index):
    draw_wind_dcm(ax, side, plot_dict, cosmetics, index)
    return ax


def draw_wind_dcm(ax, side, plot_dict, cosmetics, index):
    dcm_colors = cosmetics['trajectory']['dcm_colors']

    variables_si = viz_tools.assemble_variable_slice_from_interpolated_data(plot_dict, index)

    architecture = plot_dict['architecture']
    for parent in architecture.layer_nodes:

        avg_radius = plot_dict['outputs']['actuator']['avg_radius' + str(parent)][0][index]
        visibility_scaling = avg_radius

        wind_dcm = get_wind_dcm_var(variables_si, parent)
        uzero_hat = wind_dcm[:, 0]
        vzero_hat = wind_dcm[:, 1]
        wzero_hat = wind_dcm[:, 2]

        ehat_dict = {'x': uzero_hat,
                     'y': vzero_hat,
                     'z': wzero_hat}

        x_start = []
        for dim in range(3):
            local = plot_dict['outputs']['actuator']['center' + str(parent)][dim][index]
            x_start = cas.vertcat(x_start, local)

        for vec_name, vec_ehat in ehat_dict.items():
            x_end = x_start + visibility_scaling * vec_ehat

            color = dcm_colors[vec_name]
            viz_tools.basic_draw(ax, side, color=color, x_start=x_start, x_end=x_end, linestyle='--')

    return None
