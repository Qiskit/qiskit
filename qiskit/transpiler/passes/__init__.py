# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===================================================
Transpiler Passes (:mod:`qiskit.transpiler.passes`)
===================================================

.. currentmodule:: qiskit.transpiler.passes

Layout Selection (Placement)
============================

.. autosummary::
   :toctree: ../stubs/

   SetLayout
   TrivialLayout
   DenseLayout
   NoiseAdaptiveLayout
   SabreLayout
   CSPLayout
   ApplyLayout
   Layout2qDistance
   EnlargeWithAncilla
   FullAncillaAllocation

Routing
=======

.. autosummary::
   :toctree: ../stubs/

   BasicSwap
   LookaheadSwap
   StochasticSwap
   SabreSwap
   BIPMapping

Basis Change
============

.. autosummary::
   :toctree: ../stubs/

   Unroller
   Unroll3qOrMore
   Decompose
   UnrollCustomDefinitions
   BasisTranslator

Optimizations
=============

.. autosummary::
   :toctree: ../stubs/

   Optimize1qGates
   Optimize1qGatesDecomposition
   Collect2qBlocks
   ConsolidateBlocks
   CXCancellation
   CommutationAnalysis
   CommutativeCancellation
   Optimize1qGatesSimpleCommutation
   RemoveDiagonalGatesBeforeMeasure
   RemoveResetInZeroState
   CrosstalkAdaptiveSchedule
   TemplateOptimization

Calibration
=============

.. autosummary::
   :toctree: ../stubs/

   PulseGates
   RZXCalibrationBuilder
   RZXCalibrationBuilderNoEcho

Scheduling
=============

.. autosummary::
   :toctree: ../stubs/

   TimeUnitConversion
   ALAPSchedule
   ASAPSchedule
   DynamicalDecoupling
   AlignMeasures
   ValidatePulseGates

Circuit Analysis
================

.. autosummary::
   :toctree: ../stubs/

   Width
   Depth
   Size
   CountOps
   CountOpsLongestPath
   NumTensorFactors
   DAGLongestPath

Synthesis
=============

.. autosummary::
   :toctree: ../stubs/

   UnitarySynthesis

Additional Passes
=================

.. autosummary::
   :toctree: ../stubs/

   CheckMap
   CheckCXDirection
   CheckGateDirection
   CXDirection
   GateDirection
   MergeAdjacentBarriers
   RemoveBarriers
   BarrierBeforeFinalMeasurements
   RemoveFinalMeasurements
   DAGFixedPoint
   FixedPoint
   ContainsInstruction
   GatesInBasis
"""

# layout selection (placement)
from .layout import SetLayout
from .layout import TrivialLayout
from .layout import DenseLayout
from .layout import NoiseAdaptiveLayout
from .layout import SabreLayout
from .layout import CSPLayout
from .layout import ApplyLayout
from .layout import Layout2qDistance
from .layout import EnlargeWithAncilla
from .layout import FullAncillaAllocation

# routing
from .routing import BasicSwap
from .routing import LayoutTransformation
from .routing import LookaheadSwap
from .routing import StochasticSwap
from .routing import SabreSwap
from .routing import BIPMapping

# basis change
from .basis import Decompose
from .basis import Unroller
from .basis import UnrollCustomDefinitions
from .basis import Unroll3qOrMore
from .basis import BasisTranslator

# optimization
from .optimization import Optimize1qGates
from .optimization import Optimize1qGatesDecomposition
from .optimization import Collect2qBlocks
from .optimization import CollectMultiQBlocks
from .optimization import ConsolidateBlocks
from .optimization import CommutationAnalysis
from .optimization import CommutativeCancellation
from .optimization import CXCancellation
from .optimization import Optimize1qGatesSimpleCommutation
from .optimization import OptimizeSwapBeforeMeasure
from .optimization import RemoveResetInZeroState
from .optimization import RemoveDiagonalGatesBeforeMeasure
from .optimization import CrosstalkAdaptiveSchedule
from .optimization import HoareOptimizer
from .optimization import TemplateOptimization
from .optimization import InverseCancellation

# circuit analysis
from .analysis import ResourceEstimation
from .analysis import Depth
from .analysis import Size
from .analysis import Width
from .analysis import CountOps
from .analysis import CountOpsLongestPath
from .analysis import NumTensorFactors
from .analysis import DAGLongestPath

# synthesis
from .synthesis import UnitarySynthesis
from .synthesis import unitary_synthesis_plugin_names

# calibration
from .calibration import PulseGates
from .calibration import RZXCalibrationBuilder
from .calibration import RZXCalibrationBuilderNoEcho

# circuit scheduling
from .scheduling import TimeUnitConversion
from .scheduling import ALAPSchedule
from .scheduling import ASAPSchedule
from .scheduling import DynamicalDecoupling
from .scheduling import AlignMeasures
from .scheduling import ValidatePulseGates

# additional utility passes
from .utils import CheckMap
from .utils import CheckCXDirection  # Deprecated
from .utils import CXDirection  # Deprecated
from .utils import CheckGateDirection
from .utils import GateDirection
from .utils import BarrierBeforeFinalMeasurements
from .utils import RemoveFinalMeasurements
from .utils import MergeAdjacentBarriers
from .utils import DAGFixedPoint
from .utils import FixedPoint
from .utils import Error
from .utils import RemoveBarriers
from .utils import ContainsInstruction
from .utils import GatesInBasis
