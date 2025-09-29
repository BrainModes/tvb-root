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
from tvb.simulator.integrators import HeunDeterministic
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


class TestPolynomialModel(BaseTestCase):

    ORDER = 3
    Ps = numpy.ones((ORDER+1, ))
    N_MODES = 2

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

    def _run_for_order(self, order, pmode="fix"):
        sim = Simulator()
        sim.connectivity = Connectivity.from_file()
        sim.connectivity.configure()
        if pmode == "fix":
            p = self.Ps[:order+1]
            pdiag = p
        else:
            Nregs = sim.connectivity.number_of_regions
            p = numpy.random.normal(size=(Nregs, Nregs, self.N_MODES, order + 1))
            pdiag = numpy.einsum("jj...->j...", p)
            assert pdiag.shape == (Nregs, self.N_MODES, order + 1)
            jj = numpy.arange(Nregs).astype("i")
            p[jj, jj] = 0.0
            assert numpy.all(p[jj, jj] == 0)
        sim.coupling = PolynomialCoupling(p=p)
        sim.model = Polynomial(p=pdiag)
        sim.model.number_of_modes = self.N_MODES
        sim.model.configure()
        sim.integrator.dt = 0.1
        sim.simulation_length = 0.2
        sim.initial_conditions = numpy.ones((1, order, sim.connectivity.number_of_regions, sim.model.number_of_modes))
        sim.monitors = (Raw(), )
        sim.configure()
        self._configuration(sim.model, sim.coupling, order)
        self._polyval(sim.initial_conditions[0], sim.model.p[:, :, 1:])
        results = sim.run()
        assert results[0][1].shape == (2, 1, sim.connectivity.number_of_regions, sim.model.number_of_modes)

    def test_polynomial_model(self):
        for order in range(1, self.ORDER+1):
            self._run_for_order(order, pmode="fix")
            self._run_for_order(order, pmode="fullrandom")
