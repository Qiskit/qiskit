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
   Commuting2qGateRouter
   LayoutTransformation
   LookaheadSwap
   SabreSwap
   StarPreRouting

Basis Change
============

.. autosummary::
   :toctree: ../stubs/

   BasisTranslator
   Decompose
   TranslateParameterizedGates
   Unroll3qOrMore
   UnrollCustomDefinitions

Optimizations
=============

.. autosummary::
   :toctree: ../stubs/

   Collect1qRuns
   Collect2qBlocks
   CollectAndCollapse
   CollectCliffords
   CollectLinearFunctions
   CollectMultiQBlocks
   CommutationAnalysis
   CommutativeCancellation
   CommutativeInverseCancellation
   ConsolidateBlocks
   ContractIdleWiresInControlFlow
   ElidePermutations
   HoareOptimizer
   InverseCancellation
   Optimize1qGates
   Optimize1qGatesDecomposition
   Optimize1qGatesSimpleCommutation
   OptimizeAnnotated
   OptimizeCliffordT
   OptimizeCliffords
   OptimizeSwapBeforeMeasure
   RemoveDiagonalGatesBeforeMeasure
   RemoveFinalReset
   RemoveIdentityEquivalent
   RemoveResetInZeroState
   ResetAfterMeasureSimplification
   Split2QUnitaries
   TemplateOptimization

Scheduling
=============

.. autosummary::
   :toctree: ../stubs/

   ALAPScheduleAnalysis
   ASAPScheduleAnalysis
   ConstrainedReschedule
   ContextAwareDynamicalDecoupling
   InstructionDurationCheck
   PadDelay
   PadDynamicalDecoupling
   SetIOLatency
   TimeUnitConversion

Circuit Analysis
================

.. autosummary::
   :toctree: ../stubs/

   CountOps
   CountOpsLongestPath
   DAGLongestPath
   Depth
   NumTensorFactors
   ResourceEstimation
   Size
   Width

Synthesis
=========

The synthesis transpiler plugin documentation can be found in the
:mod:`qiskit.transpiler.passes.synthesis.plugin` page.

.. autosummary::
   :toctree: ../stubs/

   HLSConfig
   HighLevelSynthesis
   LinearFunctionsToPermutations
   SolovayKitaev
   UnitarySynthesis

Post Layout
===========

These are post qubit selection.

.. autosummary::
   :toctree: ../stubs/

   VF2PostLayout

Additional Passes
=================

.. autosummary::
   :toctree: ../stubs/

   BarrierBeforeFinalMeasurements
   CheckGateDirection
   CheckMap
   ContainsInstruction
   DAGFixedPoint
   Error
   FilterOpNodes
   FixedPoint
   GateDirection
   GatesInBasis
   MergeAdjacentBarriers
   MinimumPoint
   RemoveBarriers
   RemoveFinalMeasurements
   UnrollForLoops
"""

# layout selection (placement)
from .layout import SetLayout
from .layout import TrivialLayout
from .layout import DenseLayout
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
from .routing import Commuting2qGateRouter
from .routing import LayoutTransformation
from .routing import LookaheadSwap
from .routing import SabreSwap
from .routing import StarPreRouting

# basis change
from .basis import BasisTranslator
from .basis import Decompose
from .basis import TranslateParameterizedGates
from .basis import Unroll3qOrMore
from .basis import UnrollCustomDefinitions

# optimization
from .optimization import Collect1qRuns
from .optimization import Collect2qBlocks
from .optimization import CollectAndCollapse
from .optimization import CollectCliffords
from .optimization import CollectLinearFunctions
from .optimization import CollectMultiQBlocks
from .optimization import CommutationAnalysis
from .optimization import CommutativeCancellation
from .optimization import CommutativeInverseCancellation
from .optimization import ConsolidateBlocks
from .optimization import ContractIdleWiresInControlFlow
from .optimization import ElidePermutations
from .optimization import HoareOptimizer
from .optimization import InverseCancellation
from .optimization import Optimize1qGates
from .optimization import Optimize1qGatesDecomposition
from .optimization import Optimize1qGatesSimpleCommutation
from .optimization import OptimizeAnnotated
from .optimization import OptimizeCliffordT
from .optimization import OptimizeCliffords
from .optimization import OptimizeSwapBeforeMeasure
from .optimization import RemoveDiagonalGatesBeforeMeasure
from .optimization import RemoveFinalReset
from .optimization import RemoveIdentityEquivalent
from .optimization import RemoveResetInZeroState
from .optimization import ResetAfterMeasureSimplification
from .optimization import Split2QUnitaries
from .optimization import TemplateOptimization

# circuit analysis
from .analysis import CountOps
from .analysis import CountOpsLongestPath
from .analysis import DAGLongestPath
from .analysis import Depth
from .analysis import NumTensorFactors
from .analysis import ResourceEstimation
from .analysis import Size
from .analysis import Width

# synthesis
from .synthesis import AQCSynthesisPlugin
from .synthesis import CliffordUnitarySynthesis
from .synthesis import HLSConfig
from .synthesis import HighLevelSynthesis
from .synthesis import LinearFunctionsToPermutations
from .synthesis import SolovayKitaev
from .synthesis import SolovayKitaevSynthesis
from .synthesis import UnitarySynthesis
from .synthesis import unitary_synthesis_plugin_names

# circuit scheduling
from .scheduling import ALAPScheduleAnalysis
from .scheduling import ASAPScheduleAnalysis
from .scheduling import ConstrainedReschedule
from .scheduling import ContextAwareDynamicalDecoupling
from .scheduling import InstructionDurationCheck
from .scheduling import PadDelay
from .scheduling import PadDynamicalDecoupling
from .scheduling import SetIOLatency
from .scheduling import TimeUnitConversion

# additional utility passes
from .utils import BarrierBeforeFinalMeasurements
from .utils import CheckGateDirection
from .utils import CheckMap
from .utils import ContainsInstruction
from .utils import DAGFixedPoint
from .utils import Error
from .utils import FilterOpNodes
from .utils import FixedPoint
from .utils import GateDirection
from .utils import GatesInBasis
from .utils import MergeAdjacentBarriers
from .utils import MinimumPoint
from .utils import RemoveBarriers
from .utils import RemoveFinalMeasurements
from .utils import UnrollForLoops
