import re
from circuit_layer_analyzer import gate_qubit_tuples_of_circuit_as_layers, gate_qubit_tuples_of_circuit_as_one
from plot_quantum_circuit import enumerate_gates

start_x = 60
start_y = 20
gap_x = 30
gap_y = 30
text_height = 10

padding_bottom = 24
padding_right = 24

box_side = 24
circle_r = 10
dot_r = 3
stroke_width = 2

text_q = '<text step="{}" x="{}" y="{}" fill="black" font-family="monospace">{}</text>'
text_g = '<text step="{}" x="{}" y="{}" fill="black" font-family="HelveticaNeue-light">{}</text>'
line = '<line step="{}" x1="{}" y1="{}" x2="{}" y2="{}" fill="black" stroke="black" stroke-width="{}"></line>'
rect = '<rect step="{}" x="{}" y="{}" width="{}" height="{}" style="fill:{};"></rect>'
dot = '<circle step="{}" cx="{}" cy="{}" r="{}" stroke="black" stroke-width="{}" fill="black"></circle>'
circle = '<circle step="{}" cx="{}" cy="{}" r="{}" stroke="black" stroke-width="{}" fill="none"></circle>'

# if textstr in ['ID', 'id']:
#             fillcolor = mcolors.cnames['gold']
#         elif textstr in ['X', 'Y', 'Z', 'x', 'y', 'z']:
#             fillcolor = mcolors.cnames['lime']
#         elif textstr in ['H', 'S', 'SDG', 'h', 's', 'sdg']:
#             fillcolor = mcolors.cnames['deepskyblue']
#         elif textstr in ['T', 'TDG', 't', 'tdg']:
#             fillcolor = mcolors.cnames['tomato']
#         else:
#             fillcolor = 'w'

gate_color = {
    'h': 'rgb(0, 193, 240)',
    's': 'rgb(0, 193, 240)',
    'sdg': 'rgb(0, 193, 240)',
    'x': 'rgb(164, 202, 69)',
    'y': 'rgb(164, 202, 69)',
    'z': 'rgb(164, 202, 69)',
    't': 'rgb(255, 105, 180)',
    'tdg': 'rgb(255, 105, 180)',
    'measure': 'rgb(245, 116, 169)',
    'H': 'rgb(0, 193, 240)',
    'S': 'rgb(0, 193, 240)',
    'SDG': 'rgb(0, 193, 240)',
    'X': 'rgb(164, 202, 69)',
    'Y': 'rgb(164, 202, 69)',
    'Z': 'rgb(164, 202, 69)',
    'T': 'rgb(255, 105, 180)',
    'TDG': 'rgb(255, 105, 180)',
    'MEASURE': 'rgb(245, 116, 169)'
}

class SVGenWithCircuit:
    def __init__(self, circuit):
        self._circuit = circuit
        self._steps = 0
        self._qubit_y = {}
        self._cur_step_qubits = {}

        self._max_x = None
        self._max_y = None





    def generate_svg(self):
        qubits = {}

        schedule, global_qubit_inits, _ = gate_qubit_tuples_of_circuit_as_layers(self._circuit)

        #header:
        labels = []
        for i,gate in enumerate_gates(schedule,schedule=True):
            for label in gate[1:]:
                if label not in labels:
                    labels.append(label)

        # declare:
        declarations = []
        i = 0
        for name in labels:
            idx = 0
            y = i * gap_y + start_y
            declarations.append(text_q.format(self._steps, 10, y + text_height / 2, name))
            declarations.append(line.format(self._steps, start_x + box_side, y, '{}', y, stroke_width))
            i = i + 1

        self._steps += 1

        # for layer in schedule_list_of_layers:
        # list_per_layer: [('H', 'var0'), ('H', 'var1'), ('H', 'var2'), ('X', 'conj0'), ('X', 'conj1'), ('X', 'conj2')]
        progress = []
        for layer in schedule:
            for item in layer: #('H', 'var0')
                if item[0].lower() == "ccx":
                    gate = item[0]
                    aname = item[1]
                    ai = labels.index(aname)
                    ay = ai * gap_y + start_y

                    bname = item[2]
                    bi = labels.index(bname)
                    by = bi * gap_y + start_y

                    cname = item[3]
                    ci = labels.index(cname)
                    cy = ci * gap_y + start_y

                    progress.append(line.format(self._steps, start_x + self._steps * gap_x + box_side / 2, ay, start_x + self._steps * gap_x + box_side / 2, by, stroke_width))
                    progress.append(line.format(self._steps, start_x + self._steps * gap_x + box_side / 2, by, start_x + self._steps * gap_x + box_side / 2, cy, stroke_width))
                    progress.append(line.format(self._steps, start_x + self._steps * gap_x + box_side / 2, ay, start_x + self._steps * gap_x + box_side / 2, cy, stroke_width))
                    progress.append(dot.format(self._steps, start_x + self._steps * gap_x + box_side / 2, ay, dot_r, stroke_width))
                    progress.append(dot.format(self._steps, start_x + self._steps * gap_x + box_side / 2, by, dot_r, stroke_width))
                    progress.append(circle.format(self._steps, start_x + self._steps * gap_x + box_side / 2, cy, circle_r, stroke_width))
                    progress.append(line.format(self._steps, start_x + self._steps * gap_x + 2, cy, start_x + self._steps * gap_x + 2 * circle_r + 2, cy, stroke_width))
                    progress.append(line.format(self._steps, start_x + self._steps * gap_x + circle_r + 2, cy - circle_r, start_x + self._steps * gap_x + circle_r + 2, cy + circle_r, stroke_width))


                elif item[0].lower() == "cx" or item[0].lower() == "cy" or item[0].lower() == "cz": # take bc from abc in ccx
                    gate = item[0]
                    bname = item[1]
                    bi = labels.index(bname)
                    by = bi * gap_y + start_y

                    cname = item[2]
                    ci = labels.index(cname)
                    cy = ci * gap_y + start_y

                    progress.append(line.format(self._steps, start_x + self._steps * gap_x + box_side / 2, by, start_x + self._steps * gap_x + box_side / 2, cy, stroke_width))
                    progress.append(dot.format(self._steps, start_x + self._steps * gap_x + box_side / 2, by, dot_r, stroke_width))
                    progress.append(circle.format(self._steps, start_x + self._steps * gap_x + box_side / 2, cy, circle_r, stroke_width))
                    progress.append(line.format(self._steps, start_x + self._steps * gap_x + 2, cy, start_x + self._steps * gap_x + 2 * circle_r + 2, cy, stroke_width))
                    progress.append(line.format(self._steps, start_x + self._steps * gap_x + circle_r + 2, cy - circle_r, start_x + self._steps * gap_x + circle_r + 2, cy + circle_r, stroke_width))

                    if item[0].lower() == "cy" or item[0].lower() == "cz":
                        progress.append(text_g.format(self._steps, start_x + self._steps * gap_x + box_side / 8, cy + circle_r/2, gate.upper()))

                else:
                    gate = item[0]
                    name = item[1]
                    i = labels.index(name)
                    y = i * gap_y + start_y
                    progress.append(rect.format(self._steps, start_x + self._steps * gap_x, y - box_side / 2, box_side, box_side, gate_color[gate]))

                    gate_text = gate[0]
                    if gate == 'sdg' or gate == 'SDG':
                        gate_text = gate_text + '\''
                    progress.append(text_g.format(self._steps, start_x + self._steps * gap_x + box_side / 4, y + box_side / 4, gate_text.upper()))

            self._steps += 1

        self._max_x = start_x + self._steps * gap_x
        self._max_y = (len(labels)-1) * gap_y + start_y

        #self._max_x = int([float(a.split('=')[1].replace('"', '')) for a in progress[-1].split() if a[0] == 'x'][0]) + box_side
        # return '<svg width="{}" height="{}">\n'.format(self._max_x + padding_right, self._max_y + padding_bottom) + '\n'.join([d.format(self._max_x) for d in declarations] + progress) + '\n</svg>'
        return '<svg id= "svginstance" class="generated-qasm-svg" width="100%" height="100%" viewBox="0 0 {} {}" preserveAspectRatio="xMaxYMax">\n'.format(self._max_x + padding_right, self._max_y + padding_bottom) \
               + '\n'.join([d.format(self._max_x) for d in declarations] + progress) + \
               '\n</svg>'


