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
        kite_radius = geom.get_vector_from_center_to_kite(model_options, variables, architecture, kite)
        aero_moment = vect_op.cross(kite_radius, aero_force)

        total_moment_aero = total_moment_aero + aero_moment

    return total_moment_aero

def get_actuator_moment_projected_ref(model_options, parameters, scaling, parent, wind, architecture):

    f_aero_scaling = model_options['scaling']['z']['f_aero']

    dcm_act, dcm_wind = actuator_flow.estimate_reference_actuator_and_wind_reference_frames(model_options, wind, scaling)
    f_hat_temp = dcm_wind[:, 0]
    r_hat_temp = dcm_act[:, 1]
    ehat_perp = dcm_act[:, 2]

    #
    #
    # altitude = model_options['wind']['estimated_operating_altitude']
    # l_t = scaling['x', 'l_t']
    # inclination = cas.asin(altitude / l_t)
    #
    # f_hat_temp = vect_op.xhat_dm()
    # r_hat_temp = -cas.sin(inclination) * vect_op.xhat_dm() + cas.cos(inclination) * vect_op.zhat_dm()
    # ehat_perp = vect_op.normed_cross(f_hat_temp, r_hat_temp)

    aero_force = f_aero_scaling * f_hat_temp
    radius_ref = actuator_geom.get_radius_ref(model_options, parameters)
    kite_radius = radius_ref * r_hat_temp

    aero_moment = vect_op.cross(kite_radius, aero_force)
    number_of_children = architecture.get_number_children(parent)
    total_moment_aero = aero_moment * number_of_children
    moment_projected = vect_op.smooth_abs(cas.mtimes(total_moment_aero.T, ehat_perp))

    return moment_projected


def get_actuator_thrust_val(variables, outputs, parent, architecture):

    total_force_aero = get_actuator_force(outputs, parent, architecture)
    n_hat_var = actuator_system.get_actuator_vector_unit_var(variables, 'n', parent)
    thrust = cas.mtimes(total_force_aero.T, n_hat_var)

    return thrust

def get_actuator_thrust_var(variables_si, parent):
    var_type = 'z'
    prefix = actuator_system.get_actuator_var_name_prefix()
    var_name = prefix + 'thrust' + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var


def get_thrust_cstr(variables, outputs, parent, architecture, scaling):
    thrust_var = get_actuator_thrust_var(variables, parent)
    thrust_val = get_actuator_thrust_val(variables, outputs, parent, architecture)

    resi_si = thrust_val - thrust_var

    var_type = 'z'
    prefix = actuator_system.get_actuator_var_name_prefix()
    var_name = prefix + 'thrust' + str(parent)
    resi_scaled = struct_op.var_si_to_scaled(var_type, var_name, resi_si, scaling)

    name = 'actuator_thrust_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi_scaled,
                              name=name,
                              cstr_type='eq')
    return cstr
