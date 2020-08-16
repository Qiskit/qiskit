# -*- coding: utf-8 -*-

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

r"""

"""

from typing import Optional, List

import numpy as np

from qiskit import circuit
from qiskit.visualization.timeline import drawer_style, drawing_objects, events, types


class DrawDataContainer:
    """Data container for drawing objects."""

    def __init__(self):
        """Create new data container."""

        # drawing objects
        self.drawings = []

        # boundary box
        self.bbox_top = 0
        self.bbox_bottom = 0
        self.bbox_right = 0
        self.bbox_left = 0

        # vertical offset of bits
        self.bit_offsets = {}

        # events
        self.events = {}

        # bits
        self.bits = []

    def load_program(self,
                     scheduled_circuit: circuit.QuantumCircuit,
                     inst_durations: events.InstructionDurations):
        """Load quantum circuit and create drawing object..

        Args:
            scheduled_circuit: Scheduled circuit object to draw.
            inst_durations: Table of gate lengths.
        """
        self.bits = scheduled_circuit.qubits + scheduled_circuit.clbits

        for bit in self.bits:
            self.events[bit] = events.BitEvents.load_program(scheduled_circuit=scheduled_circuit,
                                                             inst_durations=inst_durations,
                                                             bit=bit)
        # update bbox
        self.set_time_range(0, scheduled_circuit.duration)

        # generate drawing objects associated with the events
        for bit, bit_events in self.events.items():
            # create objects associated with gates
            for gen in drawer_style['generator.gates']:
                for drawing in sum(list(map(gen, bit_events.gates())), []):
                    self._add_drawing(drawing)
            # create objects associated with barriers
            for gen in drawer_style['generator.barriers']:
                for drawing in sum(list(map(gen, bit_events.barriers())), []):
                    self._add_drawing(drawing)
            # create objects associated with bit links
            for gen in drawer_style['generator.bit_links']:
                for drawing in sum(list(map(gen, bit_events.bit_links())), []):
                    self._add_drawing(drawing)

        # create objects associated with bits
        for gen in drawer_style['generator.bits']:
            for drawing in sum(list(map(gen, self.bits)), []):
                self._add_drawing(drawing)

    def set_time_range(self,
                       t_start: int,
                       t_end: int):
        """Set time range to draw.

        Args:
            t_start: Left boundary of drawing in units of cycle time.
            t_end: Right boundary of drawing in units of cycle time.
        """
        duration = t_end - t_start

        self.bbox_left = t_start - int(duration * drawer_style['formatter.margin.left_percent'])
        self.bbox_right = t_end + int(duration * drawer_style['formatter.margin.right_percent'])

    def update_preference(self,
                          visible_bits: Optional[List[types.Bits]] = None):
        """"""
        active_bits = self._sort_bits(visible_bits)
        self.bit_offsets = {bit: 0 for bit in active_bits}

        # update bit offset coordinates
        y0 = -drawer_style['formatter.margin.top']
        y_interval = drawer_style['formatter.margin.interval']
        for bit in active_bits:
            offset = y0 - 0.5
            self.bit_offsets[bit] = offset
            y0 = offset - (0.5 + y_interval)

        # update visible option
        for drawing in self.drawings:
            if drawing.data_type == types.DrawingLine.BIT_LINK:
                # bit link
                n_points = 0
                for bit in drawing.bits:
                    if bit in active_bits:
                        n_points += 1
                if n_points > 1:
                    drawing.visible = True
                else:
                    drawing.visible = False
            else:
                # standard bit associated object
                if drawing.bit in active_bits:
                    drawing.visible = True
                else:
                    drawing.visible = False

        # update bbox
        self.bbox_top = 0
        self.bbox_bottom = y0 - (drawer_style['formatter.margin.bottom'] - y_interval)

        # update offset of bit links
        self._check_link_overlap()

    def _sort_bits(self,
                   visible_bits: List[types.Bits]) -> List[types.Bits]:
        """Helper method to initialize and sort bit order.

        Args:
            visible_bits: List of bits to draw.
        """
        bit_arange = drawer_style['layout.bit_arange']

        if visible_bits is None:
            bits = []
            for bit in self.bits:
                # remove classical bit
                if isinstance(bit, circuit.Clbit) and \
                        not drawer_style['formatter.control.show_clbits']:
                    continue
                # remove idle bit
                if self.events[bit].is_empty() and \
                        not drawer_style['formatter.control.show_idle']:
                    continue

                bits.append(bit)
        else:
            bits = visible_bits

        if len(bits) > 1:
            return bit_arange(bits)
        else:
            return bits

    def _check_link_overlap(self):
        """Helper method to check overlap of bit links.

        This method dynamically shifts horizontal position of links if they are overlapped.
        """
        allowed_overlap = drawer_style['formatter.margin.link_interval_dt']

        # extract active links
        links = []
        for drawing in self.drawings:
            if drawing.data_type == types.DrawingLine.BIT_LINK and drawing.visible:
                links.append(drawing)

        # return y coordinates
        def y_coords(link: drawing_objects.BitLinkData):
            return np.array([self.bit_offsets.get(bit, None) for bit in link.bits])

        # group overlapped links
        overlapped_group = []
        while len(links) > 0:
            ref_link = links.pop()
            overlaps = [ref_link]
            for ind in reversed(range(len(links))):
                trg_link = links[ind]
                if np.abs(ref_link.x - trg_link.x) < allowed_overlap:
                    y0s = y_coords(ref_link)
                    y1s = y_coords(trg_link)
                    v1 = np.nanmin(y0s) - np.nanmin(y1s)
                    v2 = np.nanmax(y0s) - np.nanmax(y1s)
                    v3 = np.nanmin(y0s) - np.nanmax(y1s)
                    v4 = np.nanmax(y0s) - np.nanmin(y1s)
                    if not (v1 * v2 > 0 and v3 * v4 > 0):
                        overlaps.append(links.pop(ind))
            overlapped_group.append(overlaps)

        # renew horizontal offset
        for overlaps in overlapped_group:
            if len(overlaps) > 1:
                xpos_mean = np.mean([link.x for link in overlaps])

                # sort link by y position
                sorted_links = sorted(overlaps,
                                      key=lambda x: np.nanmin(y_coords(x)))
                x0 = xpos_mean - 0.5 * allowed_overlap * (len(overlaps) - 1)
                for ind, link in enumerate(sorted_links):
                    new_x = x0 + ind * allowed_overlap
                    link.offset = new_x - link.x
                    self._add_drawing(link)
            else:
                link = overlaps[0]
                link.offset = 0
                self._add_drawing(link)

    def _add_drawing(self,
                     drawing: drawing_objects.ElementaryData):
        """Helper method to add drawing object to container.

        If the input drawing object already exists in the data container,
        this method just replaces the existing object with the input object
        instead of adding it to the list.

        Args:
            drawing: Drawing object to add to the container.
        """
        if drawing in self.drawings:
            ind = self.drawings.index(drawing)
            self.drawings[ind] = drawing
        else:
            self.drawings.append(drawing)
