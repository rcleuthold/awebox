#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
currently for untilted rotor with no tip correction factor.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-19
- edit: jochem de schutter, alu-fr 2019
- edit: rcleuthold, alu-fr, 2026
'''

import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.geometry_dir.geometry as geom
import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom
import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.actuator_dir.system as actuator_system
import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.constraint_operations as cstr_op


def get_actuator_force(outputs, parent, architecture):

    children = architecture.kites_map[parent]

    total_force_aero = cas.DM.zeros((3, 1))
    for kite in children:
        aero_force = outputs['aerodynamics']['f_aero_earth' + str(kite)]
        total_force_aero = total_force_aero + aero_force

    return total_force_aero

def get_actuator_moment(model_options, variables, outputs, parent, architecture):

    children = architecture.kites_map[parent]

    total_moment_aero = np.zeros((3, 1))
    for kite in children:
        aero_force = outputs['aerodynamics']['f_aero_earth' + str(kite)]
        vector_from_center_to_kite = geom.get_vector_from_center_to_kite(model_options, variables, architecture, kite)
        aero_moment = calculate_actuator_moment_from_force_and_arm(aero_force, vector_from_center_to_kite)

        total_moment_aero = total_moment_aero + aero_moment

    return total_moment_aero

def calculate_actuator_moment_from_force_and_arm(aero_force, vector_from_center_to_kite):
    return vect_op.cross(vector_from_center_to_kite, aero_force)

def get_actuator_moment_ref(model_options, parameters, scaling, parent, wind, architecture):
    number_of_children = architecture.get_number_children(parent)

    dcm_act, dcm_wind = actuator_flow.estimate_reference_actuator_and_wind_reference_frames(model_options, wind, scaling)
    xhat = vect_op.xhat_dm()
    u_hat_temp = dcm_wind[:, 0]
    n_hat_temp = dcm_act[:, 0]
    r_hat_temp = dcm_act[:, 1]

    f_aero_scaling = model_options['scaling']['z']['f_aero'] * number_of_children
    thrust_ref = get_actuator_thrust_ref(parent, scaling)

    # aero_force = thrust_ref * n_hat_temp
    aero_force = f_aero_scaling * xhat
    print('aero moment from faero scaling')
    print_op.warn_about_temporary_functionality_alteration()

    radius_ref = actuator_geom.get_radius_ref(model_options, parameters)
    vector_from_center_to_kite = radius_ref * r_hat_temp
    total_moment_aero = calculate_actuator_moment_from_force_and_arm(aero_force, vector_from_center_to_kite)

    return total_moment_aero


def get_actuator_thrust_val(variables, outputs, parent, architecture):

    total_force_aero = get_actuator_force(outputs, parent, architecture)
    n_hat_var = actuator_system.get_actuator_vector_unit_var(variables, 'n', parent)
    thrust = cas.mtimes(total_force_aero.T, n_hat_var)

    return thrust

def get_actuator_thrust_var(variables_si, parent):
    var_type = 'z'
    var_name = actuator_system.get_actuator_thrust_var_name(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var

def get_actuator_thrust_ref(parent, scaling):
    var_type = 'z'
    var_name = actuator_system.get_actuator_thrust_var_name(parent)
    return scaling[var_type, var_name]

def get_thrust_cstr(variables, outputs, parent, architecture, scaling):
    thrust_var = get_actuator_thrust_var(variables, parent)
    thrust_val = get_actuator_thrust_val(variables, outputs, parent, architecture)

    resi_si = thrust_val - thrust_var

    var_type = 'z'
    var_name = actuator_system.get_actuator_thrust_var_name(parent)
    resi_scaled = struct_op.var_si_to_scaled(var_type, var_name, resi_si, scaling)

    name = 'actuator_thrust_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi_scaled,
                              name=name,
                              cstr_type='eq')
    return cstr
