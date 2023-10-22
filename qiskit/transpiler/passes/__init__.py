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
   VF2Layout
   ApplyLayout
   Layout2qDistance
   EnlargeWithAncilla
   FullAncillaAllocation
   SabrePreLayout

Routing
=======

.. autosummary::
   :toctree: ../stubs/

   BasicSwap
   LookaheadSwap
   StochasticSwap
   SabreSwap
   Commuting2qGateRouter

Basis Change
============

.. autosummary::
   :toctree: ../stubs/

   BasisTranslator
   Decompose
   TranslateParameterizedGates
   Unroll3qOrMore
   UnrollCustomDefinitions
   Unroller

Optimizations
=============

.. autosummary::
   :toctree: ../stubs/

   Optimize1qGates
   Optimize1qGatesDecomposition
   Collect1qRuns
   Collect2qBlocks
   CollectMultiQBlocks
   CollectLinearFunctions
   CollectCliffords
   ConsolidateBlocks
   CXCancellation
   InverseCancellation
   CommutationAnalysis
   CommutativeCancellation
   CommutativeInverseCancellation
   Optimize1qGatesSimpleCommutation
   RemoveDiagonalGatesBeforeMeasure
   RemoveResetInZeroState
   CrosstalkAdaptiveSchedule
   HoareOptimizer
   TemplateOptimization
   EchoRZXWeylDecomposition
   ResetAfterMeasureSimplification
   OptimizeCliffords
   NormalizeRXAngle

Calibration
=============

.. autosummary::
   :toctree: ../stubs/

   PulseGates
   RZXCalibrationBuilder
   RZXCalibrationBuilderNoEcho
   RXCalibrationBuilder

Scheduling
=============

.. autosummary::
   :toctree: ../stubs/

   TimeUnitConversion
   ALAPScheduleAnalysis
   ASAPScheduleAnalysis
   PadDynamicalDecoupling
   PadDelay
   ConstrainedReschedule
   ValidatePulseGates
   InstructionDurationCheck
   SetIOLatency

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
=========

.. autosummary::
   :toctree: ../stubs/

   UnitarySynthesis
   LinearFunctionsSynthesis
   LinearFunctionsToPermutations
   HighLevelSynthesis
   SolovayKitaev
   SolovayKitaevSynthesis

Post Layout (Post transpile qubit selection)
============================================

.. autosummary::
   :toctree: ../stubs/

   VF2PostLayout

Additional Passes
=================

.. autosummary::
   :toctree: ../stubs/

   CheckMap
   CheckGateDirection
   GateDirection
   MergeAdjacentBarriers
   RemoveBarriers
   BarrierBeforeFinalMeasurements
   RemoveFinalMeasurements
   DAGFixedPoint
   FixedPoint
   MinimumPoint
   ContainsInstruction
   GatesInBasis
   ConvertConditionsToIfOps
   UnrollForLoops
"""

# layout selection (placement)
from .layout import SetLayout
from .layout import TrivialLayout
from .layout import DenseLayout
from .layout import NoiseAdaptiveLayout
from .layout import SabreLayout
from .layout import CSPLayout
from .layout import VF2Layout
from .layout import VF2PostLayout
from .layout import ApplyLayout
from .layout import Layout2qDistance
from .layout import EnlargeWithAncilla
from .layout import FullAncillaAllocation
from .layout import SabrePreLayout

# routing
from .routing import BasicSwap
from .routing import LayoutTransformation
from .routing import LookaheadSwap
from .routing import StochasticSwap
from .routing import SabreSwap
from .routing import Commuting2qGateRouter

# basis change
from .basis import Decompose
from .basis import Unroller
from .basis import UnrollCustomDefinitions
from .basis import Unroll3qOrMore
from .basis import BasisTranslator
from .basis import TranslateParameterizedGates

# optimization
from .optimization import Optimize1qGates
from .optimization import Optimize1qGatesDecomposition
from .optimization import Collect2qBlocks
from .optimization import Collect1qRuns
from .optimization import CollectMultiQBlocks
from .optimization import ConsolidateBlocks
from .optimization import CommutationAnalysis
from .optimization import CommutativeCancellation
from .optimization import CommutativeInverseCancellation
from .optimization import CXCancellation
from .optimization import Optimize1qGatesSimpleCommutation
from .optimization import OptimizeSwapBeforeMeasure
from .optimization import RemoveResetInZeroState
from .optimization import RemoveDiagonalGatesBeforeMeasure
from .optimization import CrosstalkAdaptiveSchedule
from .optimization import HoareOptimizer
from .optimization import TemplateOptimization
from .optimization import InverseCancellation
from .optimization import EchoRZXWeylDecomposition
from .optimization import CollectLinearFunctions
from .optimization import CollectCliffords
from .optimization import ResetAfterMeasureSimplification
from .optimization import OptimizeCliffords
from .optimization import NormalizeRXAngle

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
from .synthesis import LinearFunctionsSynthesis
from .synthesis import LinearFunctionsToPermutations
from .synthesis import HighLevelSynthesis
from .synthesis import SolovayKitaev
from .synthesis import SolovayKitaevSynthesis

# calibration
from .calibration import PulseGates
from .calibration import RZXCalibrationBuilder
from .calibration import RZXCalibrationBuilderNoEcho
from .calibration import RXCalibrationBuilder

# circuit scheduling
from .scheduling import TimeUnitConversion
from .scheduling import ALAPScheduleAnalysis
from .scheduling import ASAPScheduleAnalysis
from .scheduling import PadDynamicalDecoupling
from .scheduling import ValidatePulseGates
from .scheduling import PadDelay
from .scheduling import ConstrainedReschedule
from .scheduling import InstructionDurationCheck
from .scheduling import SetIOLatency

# additional utility passes
from .utils import CheckMap
from .utils import CheckGateDirection
from .utils import GateDirection
from .utils import BarrierBeforeFinalMeasurements
from .utils import RemoveFinalMeasurements
from .utils import MergeAdjacentBarriers
from .utils import DAGFixedPoint
from .utils import FixedPoint
from .utils import MinimumPoint
from .utils import Error
from .utils import RemoveBarriers
from .utils import ContainsInstruction
from .utils import GatesInBasis
from .utils import ConvertConditionsToIfOps
from .utils import UnrollForLoops
