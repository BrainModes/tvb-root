# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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

"""
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""
import numpy
import pytest
from tvb.basic.logger.builder import get_logger
from tvb.contrib.simulator.models.brunel_wang import BrunelWang
from tvb.contrib.simulator.models.epileptor import HMJEpileptor
from tvb.contrib.simulator.models.generic_2d_oscillator import Generic2dOscillator
from tvb.contrib.simulator.models.hindmarsh_rose import HindmarshRose
from tvb.contrib.simulator.models.jansen_rit_david import JansenRitDavid
from tvb.contrib.simulator.models.larter import Larter
from tvb.contrib.simulator.models.larter_breakspear import LarterBreakspear
from tvb.contrib.simulator.models.liley_steynross import LileySteynRoss
from tvb.contrib.simulator.models.morris_lecar import MorrisLecar
from tvb.contrib.simulator.models.wong_wang import WongWang
from tvb.contrib.simulator.models.polynomial import Polynomial, PolynomialCoupling, polyval
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.simulator import Simulator
from tvb.simulator.integrators import HeunDeterministic, EulerDeterministic
from tvb.simulator.monitors import Raw
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive


LOG = get_logger(__name__)


@pytest.mark.skip(reason="Because it opens a window for manual inspection")
class TestContribModels(BaseTestCase):

    @staticmethod
    def _show_model_figure(model_name, model_class, dt, **kwargs):
        # Do some stuff that tests or makes use of this module...
        LOG.info("Testing {} model...".format(model_name))

        # Check that the docstring examples, if there are any, are accurate.
        import doctest
        doctest.testmod()

        # Initialize Models in their default state
        model = model_class(**kwargs)

        LOG.info("Model initialized in its default state without error...")
        LOG.info("Testing phase plane interactive...")

        # Check the Phase Plane
        integrator = HeunDeterministic(dt=dt)
        ppi_fig = PhasePlaneInteractive(model=model, integrator=integrator)
        ppi_fig.show()

    def test_brunel_wang_model(self):
        self._show_model_figure("Brunel Wang", BrunelWang, 2 ** -5)

    def test_hmj_epileptor_model(self):
        self._show_model_figure("HMJEpileptor", HMJEpileptor, 2 ** -5)

    def test_generic_2d_oscillator_model(self):
        self._show_model_figure("Generic 2D Oscillator", Generic2dOscillator, 0.9)

    def test_hindmarsh_rose_model(self):
        self._show_model_figure("Hindmarsh Rose", HindmarshRose, 0.9)

    def test_jansen_rit_david_model(self):
        self._show_model_figure("Jansen Rit David", JansenRitDavid, 2 ** -5)

    def test_larter_model(self):
        self._show_model_figure("Larter", Larter, 0.9)

    def test_larter_breakspear_model(self):
        self._show_model_figure("Larter Breakspear", LarterBreakspear, 0.9, QV_max=numpy.array([1.0]),
                                QZ_max=numpy.array([1.0]), C=numpy.array([0.00]), d_V=numpy.array([0.6]),
                                aee=numpy.array([0.5]), aie=numpy.array([0.5]), gNa=numpy.array([0.0]),
                                Iext=numpy.array([0.165]), VT=numpy.array([0.65]), ani=numpy.array([0.1]))

    def test_liley_steynross(self):
        self._show_model_figure("Liley Steynross", LileySteynRoss, 0.9)

    def test_morric_lecar(self):
        self._show_model_figure("Morris Lecar", MorrisLecar, 2 ** -5)

    def test_wong_wang_model(self):
        self._show_model_figure("Wong Wang", WongWang, 2 ** -5)


def polydfun(x, p):
    order = p.shape[-1] - 1
    xp = [x]
    for io in range(2, order + 1):
        xp.append(x * xp[-1])
    xp = numpy.array(xp).T
    dx = numpy.zeros(x.shape)
    N = x.shape[0]
    for iX in range(N):
        for jX in range(N):
            # dx[iX] += p[iX, jX, 0] + numpy.sum(xp[jX] * p[iX, jX, 1:])  # explicit
            try:
                dx[iX] += p[iX, jX, 0] + numpy.einsum("j,j->...", xp[jX], p[iX, jX, 1:])
            except Exception as e:
                print(xp.shape)
                print(p.shape)
                print(xp[jX].shape)
                print(p[iX, jX, 1:].shape)
                raise e
    return dx


def polyint_Euler(sim, p):
    dt = sim.integrator.dt
    try:
        noise = numpy.sqrt(2 * sim.integrator.noise.nsig[0].item() * dt)
    except:
        noise = 0
    x = [sim.initial_conditions[-1, 0].squeeze()]
    t = [0.0]
    while t[-1] < sim.simulation_length:
        dW = noise * numpy.random.normal(size=x[-1].shape) if noise else 0.0
        x.append(x[-1] + dt * polydfun(x[-1], p) + dW)
        t.append(t[-1] + dt)
    return numpy.array(x), numpy.array(t)


class TestPolynomialModel(BaseTestCase):

    ORDER = 3
    N_MODES = 1

    def __init__(self, connectivity=None, p=None):
        if connectivity is None:
            connectivity = Connectivity.from_file()
            connectivity.weights = numpy.ones(connectivity.weights.shape).astype("f")
            numpy.fill_diagonal(connectivity.weights, 0.0)
            connectivity.tract_lengths *= 0.0
            connectivity.configure()
        self.connectivity = connectivity
        Nregs = self.connectivity.number_of_regions
        if p is None:
            p = -3.2060285572562344e-05 + \
                0.010215945850391128*numpy.random.normal(size=(Nregs, Nregs, self.N_MODES, self.ORDER + 1))
        self.p = p
        self.ORDER = self.p.shape[-1] - 1
        self.N_MODES = self.p.shape[-2]
        assert self.connectivity.weights.shape == self.p.shape[:2]
        super(TestPolynomialModel, self).__init__()

    def _polyval(self, x, p):
        assert numpy.allclose(polyval(x, p),
                              numpy.sum([p[:, :, ip] * x[ip, :, :] for ip in range(p.shape[-1])], axis=0),
                              )

    def _configuration(self, model, coupling, order):
        state_variables = ("x",)
        state_variable_mask = [True]
        non_integrated_variables = ()
        _nvar = 1
        cvar = [0]
        default_range = numpy.array([-1.0, 1.0])
        pos_default_range = numpy.array([numpy.maximum(0.0, default_range[0]), default_range[1]])
        state_variable_range = {"x": default_range}
        state_variable_dfuns = {"x": "lamda * (p0 + p1*x"}
        variables_of_interest = ('x',)
        parameter_names = ("lamda", "p", "p0", "p1")
        coupl_parameter_names = list(parameter_names[1:])
        coupl_pre_expr = "p0 + p1*x_j"
        for io in range(2, order + 1):
            sv = "x%d" % io
            state_variables += (sv,)
            non_integrated_variables += (sv,)
            state_variable_mask.append(False)
            _nvar += 1
            cvar.append(io - 1)
            p = "p%d" % io
            parameter_names += (p,)
            state_variable_dfuns["x"] += " + %s*x^%d" % (p, io)
            if numpy.mod(io, 2) == 0:
                state_variable_range.update({sv: pos_default_range ** io})
            else:
                state_variable_range.update({sv: default_range ** io})
            coupl_pre_expr += " + %s*x_j^%d" % (p, io)
            coupl_parameter_names += [p]
        state_variable_dfuns["x"] += ")"
        cvar = numpy.array(cvar)
        state_variable_mask = numpy.array(state_variable_mask)
        assert state_variables == model.state_variables
        assert numpy.all(state_variable_mask == model.state_variable_mask)
        assert non_integrated_variables == model.non_integrated_variables
        assert _nvar == model._nvar
        assert variables_of_interest == model.variables_of_interest
        for key, val in state_variable_range.items():
            assert numpy.allclose(val, model.state_variable_range[key])
        for key, val in state_variable_dfuns.items():
            assert val == model.state_variable_dfuns[key]
        assert numpy.all(cvar == model.cvar)
        assert parameter_names == model.parameter_names
        assert coupl_pre_expr == coupling.pre_expr
        assert coupl_parameter_names == coupling.parameter_names

    def _run_for_order(self, p):
        order = p.shape[-1] - 1
        sim = Simulator(connectivity=self.connectivity)
        sim.connectivity.configure()
        Nregs = sim.connectivity.number_of_regions
        pdiag = numpy.einsum("jj...->j...", p)
        assert pdiag.shape == (Nregs, self.N_MODES, order + 1)
        jj = numpy.arange(Nregs).astype("i")
        pcoupl = numpy.array(p)
        pcoupl[jj, jj] = 0.0
        assert numpy.all(pcoupl[jj, jj] == 0)
        sim.coupling = PolynomialCoupling(p=pcoupl)
        sim.model = Polynomial(p=pdiag)
        sim.model.number_of_modes = self.N_MODES
        sim.model.configure()
        sim.integrator = EulerDeterministic(dt=0.1)
        sim.simulation_length = 0.1
        sim.initial_conditions = \
            0.1*numpy.random.normal(size=(1, order, sim.connectivity.number_of_regions, sim.model.number_of_modes))
        sim.monitors = (Raw(), )
        sim.configure()
        self._configuration(sim.model, sim.coupling, order)
        self._polyval(sim.initial_conditions[0], sim.model.p[:, :, 1:])
        results = sim.run()
        assert results[0][1].shape == (1, 1, sim.connectivity.number_of_regions, sim.model.number_of_modes)
        if self.N_MODES == 1:
            res = results[0][1].squeeze()
            targres = polyint_Euler(sim, p[:, :, 0, :])[0][1:].squeeze()
            try:
                assert numpy.allclose(res, targres, rtol=1e-01, atol=1e-03)
            except Exception as e:
                print(numpy.abs(res - targres).max())
                print(res.shape)
                print(targres.shape)
                print([res.min(), res.mean(), res.max()])
                print([targres.min(), targres.mean(), targres.max()])
                print(res)
                print(targres)
                raise e

    def test_polynomial_model(self):
        for order in range(1, self.ORDER+1):
            print("order=%d" % order)
            self._run_for_order(self.p[:, :, :, :order+1])
