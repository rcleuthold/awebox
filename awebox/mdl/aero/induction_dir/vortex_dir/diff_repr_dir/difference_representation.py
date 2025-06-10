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
vortex model of awebox aerodynamics
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-21
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.induction_dir.vortex_dir.diff_repr_dir.fixing as diff_fixing
import awebox.mdl.aero.induction_dir.vortex_dir.diff_repr_dir.initialization as diff_initialization

def get_ocp_constraints(nlp_options, V, P, Xdot, Outputs, Integral_outputs, model, time_grids):
    return diff_fixing.get_constraint(nlp_options, V, P, Xdot, Outputs, Integral_outputs, model, time_grids)

def get_initialization(nlp_options, V_init_si, p_fix_num, nlp, model):
    return diff_initialization.get_initialization(nlp_options, V_init_si, p_fix_num, nlp, model)

def test(test_includes_visualization=False):
    return None


if __name__ == "__main__":
    test(test_includes_visualization=True)