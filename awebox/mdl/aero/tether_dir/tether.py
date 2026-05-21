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
tether aerodynamics model of an awe system
takes states, finds approximate total force and moment for a tether element
finds equivalent forces corresponding to the total force and moment.
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, jochem de schutter alu-fr 2020
'''
import casadi.tools as cas
import numpy as np

from awebox.logger.logger import Logger as awelogger

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.struct_operations as struct_op

import awebox.mdl.aero.tether_dir.reynolds as tether_reynolds
import awebox.mdl.aero.tether_dir.coefficients as tether_coefficients
import awebox.mdl.aero.tether_dir.element as tether_element
import awebox.mdl.aero.tether_dir.segment as tether_segment

class Tether(print_op.PrintableObject):
    def __init__(self, model_options, parameters, wind, atmos, options_object=None):
        super().__init__(options_object=options_object, name='tether')

        self.set_density(parameters)
        self.set_reynolds_number_function(atmos, parameters)
        self.set_drag_coefficient_function(model_options, parameters)
        self.set_element_drag_function(parameters, wind, atmos)
        self.set_segment_drag_function(model_options, parameters, atmos, wind)
        self.set_drag_force_distribution_function(model_options, parameters)

    def set_density(self, parameters):
        self.__density = parameters['theta0', 'tether', 'rho']

    def set_reynolds_number_function(self, atmos, parameters):
        q_local = cas.SX.sym('q_local', (3, 1))
        ua_local = cas.SX.sym('ua_local', (3, 1))
        diameter = cas.SX.sym('diameter')
        reynolds = tether_reynolds.get_reynolds_number(atmos, q_local=q_local, ua_local=ua_local, diam=diameter)
        self.__reynolds_fun = cas.Function('reynolds_fun', [q_local, ua_local, diameter, parameters], [reynolds])
        return None

    def set_drag_coefficient_function(self, model_options, parameters):
        reynolds = cas.SX.sym('reynolds')
        cd_fun, _ = tether_coefficients.get_tether_cd_fun(model_options, parameters)
        cd_found = cd_fun(reynolds, parameters)
        self.__cd_fun = cas.Function('cd_fun', [reynolds, parameters], [cd_found])
        return None

    def set_element_drag_function(self, parameters, wind, atmos):
        self.__element_drag_fun = tether_element.get_element_drag_fun(wind, atmos, parameters, cd_tether_fun=self.__cd_fun, reynolds_fun=self.__reynolds_fun)
        return None

    def set_segment_drag_function(self, model_options, parameters, atmos, wind):
        self.__segment_drag_fun = tether_segment.get_segment_drag_fun(model_options, parameters, atmos, wind, self.__element_drag_fun, self.__reynolds_fun, self.__cd_fun)
        return None

    def set_drag_force_distribution_function(self, model_options, parameters):
        self.__drag_distribution_fun = tether_segment.get_drag_force_distribution_fun(model_options, parameters, segment_drag_fun=self.__segment_drag_fun)

    def calculate_distribute_drag_forces_on_nodes(self, upper_node, variables, parameters, architecture):
        q_top, q_bottom, dq_top, dq_bottom = tether_element.get_upper_and_lower_pos_and_vel(variables, upper_node,
                                                                                            architecture)
        diam = tether_element.get_element_diameter(variables, upper_node, architecture)
        segment_info_dict = {'q_upper': q_top,
                             'q_lower': q_bottom,
                             'dq_upper': dq_top,
                             'dq_lower': dq_bottom,
                             'diameter': diam}
        segment_info_columnized = tether_element.columnize_element_info(unpacked_as_dict=segment_info_dict)
        columnized_node_forces = self.__drag_distribution_fun(segment_info_columnized, parameters)
        node_forces_dict = tether_segment.unpack_node_force_column(columnized_node_forces)
        return node_forces_dict

    def plot_upper_node_force(self, num_fig, model_options, V_opt, p_fix_num, parameters, atmos, wind, projection=vect_op.xhat_np()):
        params_at_time = struct_op.get_parameters_at_time(V_opt, p_fix_num, parameters)
        tether_segment.plot_upper_node_force(num_fig, model_options, parameters, params_at_time, self.__element_drag_fun, atmos, wind, projection=projection)
        return None

    @property
    def density(self):
        return self.__density

    @density.setter
    def density(self, value):
        awelogger.logger.warning('Cannot set density object.')

    @property
    def drag_distribution_fun(self):
        return self.__drag_distribution_fun

    @drag_distribution_fun.setter
    def drag_distribution_fun(self, value):
        awelogger.logger.warning('Cannot set drag_distribution_fun object.')

    @property
    def segment_drag_fun(self):
        return self.__segment_drag_fun

    @segment_drag_fun.setter
    def segment_drag_fun(self, value):
        awelogger.logger.warning('Cannot set segment_drag_fun object.')

    @property
    def element_drag_fun(self):
        return self.__element_drag_fun

    @element_drag_fun.setter
    def element_drag_fun(self, value):
        awelogger.logger.warning('Cannot set element_drag_fun object.')

    @property
    def cd_fun(self):
        return self.__cd_fun

    @cd_fun.setter
    def cd_fun(self, value):
        awelogger.logger.warning('Cannot set cd_fun object.')

    @property
    def reynolds_fun(self):
        return self.__reynolds_fun

    @reynolds_fun.setter
    def reynolds_fun(self, value):
        awelogger.logger.warning('Cannot set reynolds_fun object.')