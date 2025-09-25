# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Generic linear model.
"""

import numpy

from tvb.basic.neotraits._attr import Attr, NArray, Range, List

from tvb.simulator.common import simple_gen_astr
from tvb.simulator.coupling import SparseCoupling
from tvb.simulator.models.base import Model


def polyval(x, p):
    """
    :param x: A value vector of powers of x of shape (order, regions, modes)
    :param p: Polynomial coefficients of shape (regions or 1, modes or 1, order).
    :return: The evaluation of the polynomials to an output of shape (regions, modes)
    """
    # Slower, explicit version for testing:
    # return numpy.sum([p[:, :, ip] * x[ip, :, :]**(ip+1) for ip in range(p.shape[-1])], axis=0)
    return numpy.einsum("j...,...j->...", x, p)


class Polynomial(Model):

    """
    Implementing a polynomial model from
    Kashyap, A., Geenjaar, E., Bey, P. et al.
    Using an ordinary differential equation model to separate rest and task signals in fMRI.
    Nat Commun 16, 7128 (2025). https://doi.org/10.1038/s41467-025-62491-6
    We use non integrated variables to construct the x' higher polynomial orders than 1.
    By default, a polynomial order of 1 is assumed and default parameters are given for this case,
    i.e., implementing a linear coupling. A first order polynomial is the minimum acceptable order.
    """

    lamda = NArray(
        label=r":math:`\lambda`",
        default=numpy.array([0.21]),
        domain=Range(lo=-1.0, hi=1.0, step=0.001),
        doc="The decaying coefficient specifies how quickly the node's activity relaxes.")

    p = NArray(
        label=r":math:`p`",
        default=numpy.array([0.0, -1.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.001),
        doc="Polynomial coefficients of shape (..., M), where M is the polynomial order"
            "i.e., the length of the last dimension should coincide with the desired order of the polynomial."
            "The coefficients are broadcast for regions and modes if not provided explicitly. "
            "A minimum order of 1 is assumed. Parameters p0 and p1 up to first order are set from parameter p."
            "If a parameter of size 1 is given, the zero order coefficient is set by the p0 parameter (default = 0.0).")

    p0 = NArray(
        label=r":math:`p0`",
        default=numpy.array([0.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.001),
        doc="Polynomial coefficients of zero order. "
            "Set from parameter p, i.e., user input is ignored, unless p.size==1.")

    p1 = NArray(
        label=r":math:`p1`",
        default=numpy.array([-1.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.001),
        doc="Polynomial coefficients of first order. "
            "Set from parameter p, i.e., user input is ignored.")

    state_variable_range = Attr(
        field_type=dict,
        label="State Variable ranges [lo, hi]",
        default={"x": numpy.array([-1, 1])},
        doc="Range used for state variable initialization and visualization.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("x",),
        default=("x",), )

    coupling_terms = List(
        label="Coupling terms",
        # how to unpack coupling array
        default=["c"]
    )

    state_variable_dfuns = Attr(
        field_type=dict,
        label="Drift functions",
        default={
            "x": "lamda * (p0 + p1*x)",
        }
    )

    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default=tuple('lamda p p0 p1'.split()))

    state_variables = ('x',)
    non_integrated_variables = None
    _nvar = 1
    cvar = numpy.array([0], dtype=numpy.int32)

    def _configure_state_vars_funs_params(self):
        if self.p.ndim == 1:
            # Assuming same polynomial for all regions and modes:
            self.p = self.p[numpy.newaxis, numpy.newaxis, :]
        elif self.p.ndim == 2:
            # Assuming same polynomial for modes:
            self.p = self.p[:, numpy.newaxis, :]
        correct_p0 = False
        if self.p.shape[-1] == 1:
            self.p = numpy.concatenate([numpy.zeros(self.p.shape[:2]+(1, )), self.p], axis=-1)
            correct_p0 = True
        self._nvar = int(self.p.shape[-1]) - 1
        if correct_p0:
            self.p[:, :, 0] = self.p0
        else:
            self.p0 = self.p[:, :, 0]
        self.p1 = self.p[:, :, 1]
        self.cvar = numpy.arange(self._nvar, dtype=numpy.int32)
        state_variables = list(["x"])
        non_integrated_variables = list()
        default_sv_range = self.state_variable_range[self.state_variables[0]]
        state_variable_range = {"x": default_sv_range}
        pos_state_variable_range = numpy.array([numpy.maximum(0, default_sv_range[0]), default_sv_range[1]])
        state_variable_dfuns = "lamda * (p0 + p1*x"
        parameter_names = list(["lamda", "p", "p0", "p1"])
        for j in range(2, self._nvar+1):
            xj = "x%d" % j
            state_variables.append(xj)
            non_integrated_variables.append(xj)
            if numpy.mod(j, 2) == 0:
                state_variable_range[xj] = self.state_variable_range.get(xj, pos_state_variable_range**j)
            else:
                state_variable_range[xj] = self.state_variable_range.get(xj, default_sv_range**j)
            pj = "p%d" % j
            parameter_names.append(pj)
            setattr(self, pj, self.p[:, :, j])
            state_variable_dfuns += " + %s*x^%d" % (pj, j)
        state_variable_dfuns += ")"
        self.state_variables = tuple(state_variables)
        self.non_integrated_variables = tuple(non_integrated_variables)
        self.state_variable_range = state_variable_range
        self.state_variable_dfuns = {"x": state_variable_dfuns}
        self.parameter_names = tuple(parameter_names)

    def configure(self):
        self._configure_state_vars_funs_params()
        super(Polynomial, self).configure()

    def _x_powers(self, x):
        return numpy.concatenate([x] + [x**j for j in range(2, self.nvar+1)], axis=0)

    # def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
    #     return self._x_powers(state_variables[[0]])

    def update_state_variables_after_integration(self, state_variables):
        return self._x_powers(state_variables[[0]])

    def _polyval(self, x):
        return polyval(x, self.p[:, :, 1:])

    def dfun(self, state, coupling, local_coupling=0.0):
        """
        .. math::
            \dot x = \lambda (p_0 + p_1 x + ... +  p_k x^k + ... + p_{n-1} x^{n-1} + p_n x^n) + c
        """
        x = self.update_state_variables_after_integration(state)
        return self.lamda * (self.p0 + self._polyval(x)) + coupling[0] + local_coupling * x[0]


class PolynomialCoupling(SparseCoupling):
    r"""
        Implementing a polynomial model from
        Kashyap, A., Geenjaar, E., Bey, P. et al.
        Using an ordinary differential equation model to separate rest and task signals in fMRI.
        Nat Commun 16, 7128 (2025). https://doi.org/10.1038/s41467-025-62491-6:
        .. math:
         p_0 + p_1 x + ... +  p_k x^k + ... + p_{n-1} x^{n-1} + p_n x^n
        We use non integrated variables to construct the x' higher polynomial orders than 1.
        By default, a polynomial order of 1 is assumed and default parameters are given for this case,
        i.e., implementing a linear coupling. A first order polynomial is the minimum acceptable order.
    """

    p = NArray(
        label=":math:`p`",
        default=numpy.array([0.0, 1.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.001),
        doc="Polynomial coefficients of shape (..., M), where M is the polynomial order"
            "i.e., the length of the last dimension should coincide with the desired order of the polynomial."
            "The coefficients are broadcast for connections and modes if not provided explicitly. "
            "A minimum order of 1 is assumed. Parameters p0 and p1 up to first order are set from parameter p."
            "If parameter p of size 1 is given, the zero order coefficient is set by the p0 parameter (default = 0.0).")

    p0 = NArray(
        label=r":math:`p0`",
        default=numpy.array([0.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.001),
        doc="Polynomial coefficients of zero order. "
            "Set from parameter p, i.e., user input is ignored, unless p.size==1.")

    p1 = NArray(
        label=r":math:`p1`",
        default=numpy.array([-1.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.001),
        doc="Polynomial coefficients of first order. "
            "Set from parameter p, i.e., user input is ignored.")

    parameter_names = ['p', 'p0', 'p1']
    pre_expr = 'p0 + p1*x_j'
    post_expr = None  # 'a * gx + b'
    order = 1
    p_nzw = None

    def update_derived_parameters(self):
        if self.p.ndim == 1:
            # Assuming same polynomial for all connections and modes:
            self.p = self.p[numpy.newaxis,  # target region
                            numpy.newaxis,  # source region
                            numpy.newaxis,  # modes
                           :]               # polynomial order
        elif self.p.ndim == 2:
            # Assuming same polynomial for all source regions and modes:
            self.p = self.p[:,              # target region
                            numpy.newaxis,  # source region
                            numpy.newaxis,  # modes
                            :]              # polynomial order
        elif self.p.ndim == 3:
            # Assuming same polynomial for all modes:
            self.p = self.p[:,              # target region
                            :,              # source region
                            numpy.newaxis,  # modes region
                            :]              # polynomial order
        correct_p0 = False
        if self.p.shape[-1] == 1:
            self.p = numpy.concatenate([numpy.zeros(self.p.shape[:3]+(1, )), self.p], axis=-1)
            correct_p0 = True
        if correct_p0:
            self.p[:, :, :, 0] = self.p0
        else:
            self.p0 = self.p[:, :, :, 0]
        self.p1 = self.p[:, :, :, 1]
        # Now we have parameter p of polynomial coefficients of shape (regions, regions, modes, polynomial order)
        self.order = int(self.p.shape[-1]) - 1
        parameter_names = list(['p', 'p0', 'p1'])
        for j in range(2, self.p.shape[-1]):
            # Create one extra parameter per polynomial order:
            pj = "p%d" % j
            parameter_names.append(pj)
            setattr(self, pj, self.p[:, :, :, j])
            self.pre_expr += " + %s*x_j^%d" % (pj, j)
        self.parameter_names = list(parameter_names)

    def configure(self):
        """Set the right indirect call."""
        super(PolynomialCoupling, self).configure()
        self.update_derived_parameters()

    def _polyval(self, x_j):
        return polyval(x_j, self.p_nzw[:, :, 1:])

    def pre(self, x_i, x_j):
        # (history.n_cvar, history.n_nnzw, history.n_mode)
        return numpy.repeat(self.p0 + self._polyval(x_j)[numpy.newaxis],
                            x_j.shape[0], axis=0)

    def __call__(self, step, history):
        if self.p_nzw is None:
            dummy = 1.0
            if self.p.shape[0] == 1 or self.p.shape[1] == 1:
                dummy = numpy.ones(history.nnz_mask.shape + self.p.shape[2:])
            # To be executed only the first time to keep only nonzero weights' connections:
            dummy *= self.p
            self.p_nzw = dummy[history.nnz_mask, :, :]  # new shape: (nnzw, modes, polynomial order)
        return super(PolynomialCoupling, self).__call__(step, history)

    def __str__(self):
        return simple_gen_astr(self, " ".join(self.parameter_names))
