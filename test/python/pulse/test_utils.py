# # This code is part of Qiskit.
# #
# # (C) Copyright IBM 2023.
# #
# # This code is licensed under the Apache License, Version 2.0. You may
# # obtain a copy of this license in the LICENSE.txt file in the root directory
# # of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# #
# # Any modifications or derivative works of this code must retain this
# # copyright notice, and modified files need to carry a notice indicating
# # that they have been altered from the originals.

# """Test cases for the pulse util modules."""

# import collections

# from qiskit.providers.fake_provider.backends import FakeBogotaV2
# from qiskit.pulse import channels


# def test_get_qubit_channels(self):
#     """Test to get all channels operated on a given qubit."""
#     backend = FakeBogotaV2()
#     qubit = 1
#     bogota_cr_channels_map = {
#         (4, 3): 7,
#         (3, 4): 6,
#         (3, 2): 5,
#         (2, 3): 4,
#         (1, 2): 2,
#         (2, 1): 3,
#         (1, 0): 1,
#         (0, 1): 0,
#     }
#     ref = []
#     for node_qubits in bogota_cr_channels_map:
#         if qubit in node_qubits:
#             ref.append(channels.ControlChannel(bogota_cr_channels_map[node_qubits]))
#     ref.append(channels.DriveChannel(qubit))
#     ref.append(channels.MeasureChannel(qubit))
#     ref.append(channels.AcquireChannel(qubit))
#     self.assertTrue(
#         collections.Counter(backend.get_qubit_channels(qubit)) == collections.Counter(list(ref))
#     )
