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

import casadi.tools as cas
import numpy as np


import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom
import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.actuator_dir.force as actuator_force
import awebox.mdl.aero.induction_dir.actuator_dir.system as actuator_system

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op

def get_LL_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label):
    corr = actuator_flow.get_corr_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    chi = actuator_flow.get_wake_angle_chi(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    return get_LL_matrix_from_corr_and_chi(corr, chi)

def get_LL_matrix_ref(model_options,parent, scaling):
    a_ref = actuator_flow.get_a_ref(model_options)
    corr = (1. - a_ref)
    chi = 0.
    # var_type = 'z'
    # prefix = actuator_system.get_actuator_var_name_prefix()
    # var_name = prefix + 'gamma' + str(parent)
    # chi = scaling[var_type, var_name]
    return get_LL_matrix_from_corr_and_chi(corr, chi)


def get_LL_matrix_from_corr_and_chi(corr, chi):
    tanhalfchi = cas.tan(chi / 2.)
    sechalfchi = 1. / cas.cos(chi / 2.)

    LL11 = 0.25 / corr
    LL12 = 0.
    LL13 = -0.368155 * tanhalfchi
    LL21 = 0.
    LL22 = -1. * sechalfchi**2.
    LL23 = 0.
    LL31 = (0.368155 * tanhalfchi ) / corr
    LL32 = 0.
    LL33 = -1. + tanhalfchi**2.

    LL_row1 = cas.horzcat(LL11, LL12, LL13)
    LL_row2 = cas.horzcat(LL21, LL22, LL23)
    LL_row3 = cas.horzcat(LL31, LL32, LL33)
    LL_matr = cas.vertcat(LL_row1, LL_row2, LL_row3)

    return LL_matr


def get_MM_matrix():
    MM11 = 1.69765
    MM22 = 0.113177
    MM33 = 0.113177

    MM_col1 = MM11 * vect_op.xhat()
    MM_col2 = MM22 * vect_op.yhat()
    MM_col3 = MM33 * vect_op.zhat()

    MM = cas.horzcat(MM_col1, MM_col2, MM_col3)

    return MM


def get_ct_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture):
    thrust = actuator_force.get_actuator_thrust_var(variables, parent)
    area = actuator_geom.get_area_var(variables, parent)
    qzero = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)

    ct = thrust / area / qzero

    return ct


def get_actuator_moment_y_rotor(model_options, variables, outputs, parent, architecture):
    total_moment_aero = actuator_force.get_actuator_moment(model_options, variables, outputs, parent, architecture)
    y_hat_rotor = actuator_system.get_actuator_vector_unit_var(variables, 'y', parent)
    moment = cas.mtimes(total_moment_aero.T, y_hat_rotor)
    return moment

def get_actuator_moment_z_rotor(model_options, variables, outputs, parent, architecture):
    total_moment_aero = actuator_force.get_actuator_moment(model_options, variables, outputs, parent, architecture)
    z_hat_rotor = actuator_system.get_actuator_vector_unit_var(variables, 'z', parent)
    moment = cas.mtimes(total_moment_aero.T, z_hat_rotor)
    return moment




# references


def get_thrust_ref(parent, scaling):
    var_type = 'z'
    prefix = actuator_system.get_actuator_var_name_prefix()
    var_name = prefix + 'thrust' + str(parent)
    return scaling[var_type, var_name, 0]

def get_moment_ref(model_options, atmos, wind, parameters):
    reference = model_options['scaling']['z']['m_aero']
    return reference


def get_t_star(variables, parameters, parent):
    # radius / u_0 = [m] / [m/s]
    t_star_num = get_t_star_numerator_val(variables, parameters, parent)
    t_star_den = get_t_star_denominator_val(variables, parent)
    return t_star_num / t_star_den

def get_t_star_numerator_val(variables, parameters, parent):
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    bar_varrho_var = actuator_geom.get_bar_varrho_var(variables, parent)
    t_star_num = b_ref * (bar_varrho_var + 0.5)
    return t_star_num

def get_t_star_numerator_ref(model_options, parameters):
    varrho_ref = actuator_geom.get_varrho_ref(model_options)
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    t_star_num = b_ref * (varrho_ref + 0.5)
    return t_star_num

def get_t_star_denominator_val(variables_si, parent):
    uzero_mag = actuator_system.get_actuator_vector_length_var(variables_si, 'u', parent)
    t_star_den = uzero_mag
    return t_star_den

def get_t_star_denominator_ref(parent, scaling):
    t_star_den_ref = actuator_flow.get_uzero_vec_length_ref(parent, scaling)
    return t_star_den_ref


def get_c_all_components(model_options, atmos, wind, variables, parameters, outputs, parent, architecture, scaling):

    prefix = actuator_system.get_actuator_var_name_prefix()

    thrust = actuator_force.get_actuator_thrust_var(variables, parent)
    moment_y_val = get_actuator_moment_y_rotor(model_options, variables, outputs, parent, architecture)
    moment_z_val = get_actuator_moment_z_rotor(model_options, variables, outputs, parent, architecture)

    area = actuator_geom.get_area_var(variables, parent)
    qzero = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)
    u_ref = wind.get_speed_ref()
    qzero_ref = 0.5 * u_ref**2.

    bar_varrho_var = actuator_geom.get_bar_varrho_var(variables, parent)
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    radius_bar = bar_varrho_var * b_ref

    bar_varrho_ref = scaling['z', prefix + 'bar_varrho' + str(parent)]
    area_ref = scaling['z', prefix + 'area' + str(parent)]
    # area = 2 pi varrho b b -> b^2 = area / (2 pi varrho)
    wingspan_ref = vect_op.smooth_sqrt(area_ref / (2. * np.pi * bar_varrho_ref))
    radius_ref = bar_varrho_ref * wingspan_ref
    thrust_ref = scaling['z', prefix + 'thrust' + str(parent)]

    thrust_denom = area * qzero
    moment_denom = thrust_denom * radius_bar
    moment_denom_ref = area_ref * qzero_ref * radius_ref

    thrust_radius = thrust * radius_bar
    c_all = cas.vertcat(thrust_radius, moment_y_val, moment_z_val)
    c_ref = cas.DM.ones((3, 1)) * thrust_ref * radius_ref

    return c_all, moment_denom, c_ref, moment_denom_ref
