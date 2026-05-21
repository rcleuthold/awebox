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
reynolds number of a tether element
_python-3.5 / casadi-3.4.5
- edited: rachel leuthold, jochem de schutter alu-fr 2017-20
'''

import awebox.tools.vector_operations as vect_op
import casadi.tools as cas
import awebox.tools.print_operations as print_op
from awebox.mdl.wind import Wind


def get_reynolds_number(atmos, diam=None, q_upper=None, q_lower=None, q_local=None, ua_local=None, wind=None, dq_local=None):

    if diam is None:
        message = 'not enough diameter information to determine the reynolds number function'
        print_op.log_and_raise_error(message)

    if q_local is not None:
        q_average = q_local
    elif (q_upper is not None) and (q_lower is not None):
        q_average = (q_upper + q_lower) / 2.
    else:
        message = 'not enough position information to determine the reynolds number function'
        print_op.log_and_raise_error(message)

    zz = q_average[2]
    rho_infty = atmos.get_density(zz)
    mu_infty = atmos.get_viscosity(zz)

    if ua_local is not None:
        vec_ua = ua_local
    elif (wind is not None) and (dq_local is not None):
        vec_u_infty = wind.get_velocity(zz)
        vec_ua = vec_u_infty - dq_local
    else:
        message = 'not enough velocity information to determine the reynolds number function'
        print_op.log_and_raise_error(message)

    norm_ua = cas.mtimes(vec_ua.T, vec_ua) ** 0.5
    reynolds = rho_infty * norm_ua * diam / mu_infty

    return reynolds
