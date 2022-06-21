# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test schedule block subroutine reference mechanism."""

import numpy as np

from qiskit import circuit, pulse
from qiskit.pulse import builder
from qiskit.pulse.transforms import inline_subroutines
from qiskit.test import QiskitTestCase


class TestReference(QiskitTestCase):
    """Test for basic behavior of reference mechanism."""

    def test_append_schedule(self):
        """Test appending schedule without calling.

        Appended scheduels are not subroutine.
        These are directly exposed to the outer block.
        """
        with pulse.build(name="x1") as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        with pulse.build(name="y1") as sched_y1:
            builder.append_schedule(sched_x1)

        with pulse.build(name="z1") as sched_z1:
            builder.append_schedule(sched_y1)

        self.assertEqual(len(sched_z1.references), 0)

    def test_append_schedule_parameter_scope(self):
        """Test appending schedule without calling.

        Parameter in the appended schedule has the scope of outer schedule.
        """
        param = circuit.Parameter("name")

        with pulse.build(name="x1") as sched_x1:
            pulse.play(pulse.Constant(100, param, name="x1"), pulse.DriveChannel(0))

        with pulse.build(name="y1") as sched_y1:
            builder.append_schedule(sched_x1)

        with pulse.build(name="z1") as sched_z1:
            builder.append_schedule(sched_y1)

        sched_param = next(iter(sched_z1.scoped_parameters))
        self.assertEqual(sched_param.name, "z1.name")

        # object equality
        self.assertEqual(
            sched_z1.get_parameters("name", scope="z1")[0],
            param,
        )

    def test_call_schedule(self):
        """Test call schedule.

        Outer block is only aware of its inner reference.
        Nested reference is not directly exposed to the most outer block.
        """
        with pulse.build(name="x1") as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        with pulse.build(name="y1") as sched_y1:
            builder.call(sched_x1)

        with pulse.build(name="z1") as sched_z1:
            builder.call(sched_y1)

        self.assertEqual(len(sched_z1.references), 1)
        self.assertEqual(sched_z1.references["y1"], sched_y1)
        self.assertEqual(sched_y1.references["x1"], sched_x1)

    def test_call_schedule_parameter_scope(self):
        """Test call schedule.

        Parameter in the called schedule has the scope of called schedule.
        """
        param = circuit.Parameter("name")

        with pulse.build(name="x1") as sched_x1:
            pulse.play(pulse.Constant(100, param, name="x1"), pulse.DriveChannel(0))

        with pulse.build(name="y1") as sched_y1:
            builder.call(sched_x1)

        with pulse.build(name="z1") as sched_z1:
            builder.call(sched_y1)

        sched_param = next(iter(sched_z1.scoped_parameters))
        self.assertEqual(sched_param.name, "z1.y1.x1.name")

        # object equality
        self.assertEqual(
            sched_z1.get_parameters("name", scope="z1.y1.x1")[0],
            param,
        )

        # regex
        self.assertEqual(
            sched_z1.get_parameters("name", scope=r"\S.x1")[0],
            param,
        )

    def test_append_and_call_schedule(self):
        """Test call and append schedule.

        Reference is copied to the outer schedule by appending.
        Original reference is remain unchanged.
        """
        with pulse.build(name="x1") as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        with pulse.build(name="y1") as sched_y1:
            builder.call(sched_x1)

        with pulse.build(name="z1") as sched_z1:
            builder.append_schedule(sched_y1)

        self.assertEqual(len(sched_z1.references), 1)
        self.assertEqual(sched_z1.references["x1"], sched_x1)
        self.assertEqual(sched_z1.references.scope, "z1")

        # blocks[0] is sched_y1 and its reference is now point to outer block reference
        self.assertIs(sched_z1.blocks[0].references, sched_z1.references)

        # however the original program is protected to prevent unexpected mutation
        self.assertIsNot(sched_y1.references, sched_z1.references)

        # appended schedule is preserved
        self.assertEqual(len(sched_y1.references), 1)
        self.assertEqual(sched_y1.references["x1"], sched_x1)
        self.assertEqual(sched_y1.references.scope, "y1")

    def test_alignment_context(self):
        """Test nested alignment context.

        Inline alignment is identical to append_schedule operation.
        Thus scope is not newly generated.
        """
        with pulse.build(name="x1") as sched_x1:
            with pulse.align_right():
                with pulse.align_left():
                    pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        self.assertEqual(len(sched_x1.references), 0)

    def test_appending_child_block(self):
        """Test for edge case.

        User can append blocks which is an element of another schedule block.
        But this is not standard usecase.

        In this case, references may contain subroutines which don't exit in the context.
        This is because all references within the program is centralizedly
        managed in the most outer block.
        """
        with pulse.build(name="x1") as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        with pulse.build(name="y1") as sched_y1:
            pulse.play(pulse.Constant(100, 0.1, name="y1"), pulse.DriveChannel(0))

        with pulse.build(name="x2") as sched_x2:
            builder.call(sched_x1)
        self.assertEqual(sched_x2.references._keys, ["x1"])

        with pulse.build(name="y2") as sched_y2:
            builder.call(sched_y1)
        self.assertEqual(sched_y2.references._keys, ["y1"])

        with pulse.build(name="z1") as sched_z1:
            builder.append_schedule(sched_x2)
            builder.append_schedule(sched_y2)
        self.assertEqual(sched_z1.references._keys, ["x1", "y1"])

        # child block references point to its parent, i.e. sched_z1
        self.assertIs(sched_z1.blocks[0].references, sched_z1._reference_manager)
        self.assertIs(sched_z1.blocks[1].references, sched_z1._reference_manager)
        self.assertEqual(sched_z1.blocks[0].references.scope, "z1")
        self.assertEqual(sched_z1.blocks[1].references.scope, "z1")

        with pulse.build(name="z2") as sched_z2:
            # Append child block
            # The refecence of this block is sched_z1.reference thus it contains both x1 and y1.
            # However, y1 doesn't exist in the context, so only x1 should be added.

            # Usually, user will append sched_x2 directly here, rather than sched_z1.blocks[0]
            # This is why this situation is an edge case.
            builder.append_schedule(sched_z1.blocks[0])

        self.assertEqual(len(sched_z2.references), 1)
        self.assertEqual(sched_z2.references["x1"], sched_x1)

    def test_replacement(self):
        """Test nested alignment context.

        Replacing schedule block with schedule block.
        Removed block contains own reference, that should be removed with replacement.
        New block also contains reference, that should be passed to the current reference.
        """
        with pulse.build(name="x1") as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        with pulse.build(name="y1") as sched_y1:
            pulse.play(pulse.Constant(100, 0.1, name="y1"), pulse.DriveChannel(0))

        with pulse.build(name="x2") as sched_x2:
            builder.call(sched_x1)

        with pulse.build(name="y2") as sched_y2:
            builder.call(sched_y1)

        with pulse.build(name="z1") as sched_z1:
            builder.append_schedule(sched_x2)
            builder.append_schedule(sched_y2)
        self.assertEqual(len(sched_z1.references), 2)
        self.assertEqual(sched_z1.references["x1"], sched_x1)
        self.assertEqual(sched_z1.references["y1"], sched_y1)

        # Define schedule to replace
        with pulse.build(name="r1") as sched_r1:
            pulse.play(pulse.Constant(100, 0.1, name="r1"), pulse.DriveChannel(0))

        with pulse.build(name="r2") as sched_r2:
            pulse.call(sched_r1)

        sched_z2 = sched_z1.replace(sched_x2, sched_r2)
        self.assertEqual(len(sched_z2.references), 2)
        self.assertEqual(sched_z2.references["r1"], sched_r1)
        self.assertEqual(sched_z2.references["y1"], sched_y1)


class TestSubroutineWithCXGate(QiskitTestCase):
    """Test called program scope with practical example of building fully parametrized CX gate."""

    def setUp(self):
        super().setUp()

        # parameters of X pulse
        self.xp_dur = circuit.Parameter("dur")
        self.xp_amp = circuit.Parameter("amp")
        self.xp_sigma = circuit.Parameter("sigma")
        self.xp_beta = circuit.Parameter("beta")

        # amplitude of SX pulse
        self.sxp_amp = circuit.Parameter("amp")

        # parameters of CR pulse
        self.cr_dur = circuit.Parameter("dur")
        self.cr_amp = circuit.Parameter("amp")
        self.cr_sigma = circuit.Parameter("sigma")
        self.cr_risefall = circuit.Parameter("risefall")

        # channels
        self.control_ch = circuit.Parameter("ctrl")
        self.target_ch = circuit.Parameter("tgt")
        self.cr_ch = circuit.Parameter("cr")

        # echo pulse on control qubit
        with pulse.build(name="xp") as xp_sched_q0:
            pulse.play(
                pulse.Drag(
                    duration=self.xp_dur,
                    amp=self.xp_amp,
                    sigma=self.xp_sigma,
                    beta=self.xp_beta,
                ),
                channel=pulse.DriveChannel(self.control_ch),
            )
        self.xp_sched = xp_sched_q0

        # local rotation on target qubit
        with pulse.build(name="sx") as sx_sched_q1:
            pulse.play(
                pulse.Drag(
                    duration=self.xp_dur,
                    amp=self.sxp_amp,
                    sigma=self.xp_sigma,
                    beta=self.xp_beta,
                ),
                channel=pulse.DriveChannel(self.target_ch),
            )
        self.sx_sched = sx_sched_q1

        # cross resonance
        with pulse.build(name="cr") as cr_sched:
            pulse.play(
                pulse.GaussianSquare(
                    duration=self.cr_dur,
                    amp=self.cr_amp,
                    sigma=self.cr_sigma,
                    risefall_sigma_ratio=self.cr_risefall,
                ),
                channel=pulse.ControlChannel(self.cr_ch),
            )
        self.cr_sched = cr_sched

    def test_lazy_ecr(self):
        """Test lazy subroutines through ECR schedule construction."""

        with pulse.build(name="lazy_ecr") as sched:
            with pulse.align_sequential():
                pulse.call(name="cr", channels=[pulse.ControlChannel(self.cr_ch)])
                pulse.call(name="xp", channels=[pulse.DriveChannel(self.control_ch)])
                with pulse.phase_offset(np.pi, pulse.ControlChannel(self.cr_ch)):
                    pulse.call(name="cr", channels=[pulse.ControlChannel(self.cr_ch)])
                pulse.call(name="xp", channels=[pulse.DriveChannel(self.control_ch)])

        # Schedule has references
        self.assertTrue(sched.is_referenced())

        # Schedule is not schedulable because of unassigned references
        self.assertFalse(sched.is_schedulable())

        # Two references cr and xp are called
        refs = list(sched.references)
        self.assertEqual(len(refs), 2)

        # Parameters in the current scope are Parameter("cr") and Parameter("ctrl")
        params = set(p.name for p in sched.parameters)
        self.assertSetEqual(params, {"cr", "ctrl"})

        # Parameter names are scoepd
        scoped_params = set(p.name for p in sched.scoped_parameters)
        self.assertSetEqual(scoped_params, {"lazy_ecr.cr", "lazy_ecr.ctrl"})

        # Assign CR and XP schedule to the empty reference
        sched.assign_references({"cr": self.cr_sched})
        sched.assign_references({"xp": self.xp_sched})

        # Check updated references
        assigned_refs = sched.references
        self.assertEqual(assigned_refs.get("cr"), self.cr_sched)
        self.assertEqual(assigned_refs.get("xp"), self.xp_sched)

        # Parameter added from subroutines
        scoped_params = set(p.name for p in sched.scoped_parameters)
        ref_params = {
            "lazy_ecr.cr",
            "lazy_ecr.cr.amp",
            "lazy_ecr.cr.dur",
            "lazy_ecr.cr.risefall",
            "lazy_ecr.cr.sigma",
            "lazy_ecr.ctrl",
            "lazy_ecr.xp.amp",
            "lazy_ecr.xp.beta",
            "lazy_ecr.xp.dur",
            "lazy_ecr.xp.sigma",
        }
        self.assertSetEqual(scoped_params, ref_params)

        # Get parameter without scope, cr amp and xp amp are hit.
        params = sched.get_parameters(parameter_name="amp")
        self.assertEqual(len(params), 2)

        # Get parameter with scope, only xp amp
        params = sched.get_parameters(parameter_name="amp", scope="lazy_ecr.xp")
        self.assertEqual(len(params), 1)

    def test_cnot(self):
        """Integration test with CNOT schedule construction."""
        # echeod cross resonance
        with pulse.build(name="ecr", default_alignment="sequential") as ecr_sched:
            pulse.call(self.cr_sched)
            pulse.call(self.xp_sched)
            with pulse.phase_offset(np.pi, pulse.ControlChannel(self.cr_ch)):
                pulse.call(self.cr_sched)
            pulse.call(self.xp_sched)

        # cnot gate, locally equivalent to ecr
        with pulse.build(name="cx", default_alignment="sequential") as cx_sched:
            pulse.shift_phase(np.pi / 2, pulse.DriveChannel(self.control_ch))
            pulse.call(self.sx_sched)
            pulse.call(ecr_sched)

        # get parameter with scope, full scope is not needed
        xp_amp = cx_sched.get_parameters("amp", scope="xp")[0]
        self.assertEqual(self.xp_amp, xp_amp)

        # get parameter with scope, of course full scope can be specified
        xp_amp_full_scoped = cx_sched.get_parameters("amp", scope="cx.ecr.xp")[0]
        self.assertEqual(xp_amp_full_scoped, xp_amp)

        # assign parameters
        assigned_cx = cx_sched.assign_parameters(
            value_dict={
                self.cr_ch: 0,
                self.control_ch: 0,
                self.target_ch: 1,
                self.sxp_amp: 0.1,
                self.xp_amp: 0.2,
                self.xp_dur: 160,
                self.xp_sigma: 40,
                self.xp_beta: 3.0,
                self.cr_amp: 0.5,
                self.cr_dur: 800,
                self.cr_sigma: 64,
                self.cr_risefall: 2,
            },
            inplace=True,
        )
        flatten_cx = inline_subroutines(assigned_cx)

        with pulse.build(default_alignment="sequential") as ref_cx:
            # sz
            pulse.shift_phase(np.pi / 2, pulse.DriveChannel(0))
            with pulse.align_left():
                # sx
                pulse.play(
                    pulse.Drag(
                        duration=160,
                        amp=0.1,
                        sigma=40,
                        beta=3.0,
                    ),
                    channel=pulse.DriveChannel(1),
                )
            with pulse.align_sequential():
                # cr
                with pulse.align_left():
                    pulse.play(
                        pulse.GaussianSquare(
                            duration=800,
                            amp=0.5,
                            sigma=64,
                            risefall_sigma_ratio=2,
                        ),
                        channel=pulse.ControlChannel(0),
                    )
                # xp
                with pulse.align_left():
                    pulse.play(
                        pulse.Drag(
                            duration=160,
                            amp=0.2,
                            sigma=40,
                            beta=3.0,
                        ),
                        channel=pulse.DriveChannel(0),
                    )
                with pulse.phase_offset(np.pi, pulse.ControlChannel(0)):
                    # cr
                    with pulse.align_left():
                        pulse.play(
                            pulse.GaussianSquare(
                                duration=800,
                                amp=0.5,
                                sigma=64,
                                risefall_sigma_ratio=2,
                            ),
                            channel=pulse.ControlChannel(0),
                        )
                # xp
                with pulse.align_left():
                    pulse.play(
                        pulse.Drag(
                            duration=160,
                            amp=0.2,
                            sigma=40,
                            beta=3.0,
                        ),
                        channel=pulse.DriveChannel(0),
                    )

        self.assertEqual(flatten_cx, ref_cx)
