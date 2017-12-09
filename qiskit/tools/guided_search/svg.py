import re
from circuit_layer_analyzer import gate_qubit_tuples_of_circuit_as_layers, gate_qubit_tuples_of_circuit_as_one
from plot_quantum_circuit import enumerate_gates

start_x = 60
start_y = 20
gap_x = 30
gap_y = 30
text_height = 10

padding_bottom = 2
padding_right = 2

box_side = 24
circle_r = 10
dot_r = 3
stroke_width = 2

text_q = '<text step="{}" qasm="{}" x="{}" y="{}" fill="black" font-family="monospace">{}[{}]</text>'
text_g = '<text step="{}" qasm="{}" x="{}" y="{}" fill="white" font-family="HelveticaNeue-light">{}</text>'
line = '<line step="{}" qasm="{}" x1="{}" y1="{}" x2="{}" y2="{}" fill="black" stroke="black" stroke-width="{}"></line>'
rect = '<rect step="{}" qasm="{}" x="{}" y="{}" width="{}" height="{}" style="fill:{};"></rect>'
dot = '<circle step="{}" qasm="{}" cx="{}" cy="{}" r="{}" stroke="black" stroke-width="{}" fill="black"></circle>'
circle = '<circle step="{}" qasm="{}" cx="{}" cy="{}" r="{}" stroke="black" stroke-width="{}" fill="none"></circle>'

gate_color = {
    'h': 'rgb(0, 193, 240)',
    'x': 'rgb(164, 202, 69)',
    'measure': 'rgb(245, 116, 169)',
}

gate_label_x_offset = {
    'h': 1,
    'x': 2,
    'measure': 0
}



class SVGen:
    def __init__(self, qasm):
        self._qasm = qasm
        self._steps = 0
        self._qubit_y = {}
        self._cur_step_qubits = {}

        self._max_x = None
        self._max_y = None

    def _generate_svg_for_qubits_declarations(self, qubits):
        declarations = []
        ys = [y * gap_y + start_y for y in range(sum(qubits.values()))]
        self._max_y = ys[-1] + circle_r
        for name in qubits:
            self._qubit_y[name] = ys[:qubits[name]]
            ys = ys[qubits[name]:]
        for name in self._qubit_y:
            idx = 0
            for y in self._qubit_y[name]:
                declarations.append(text_q.format(self._steps, 'declaration', 10, y + text_height / 2, name, idx))
                declarations.append(line.format(self._steps, '{}[{}]'.format(name, idx), start_x + box_side, y, '{}', y, stroke_width))
                idx += 1
        self._steps += 1
        return declarations

    def _gen_svg_for_ccx(self, qasm):
        # print(qasm)
        _, a_name, a_idx, b_name, b_idx, c_name, c_idx, _ = re.split('[\s,;\[\]]*', qasm)
        a_idx, b_idx, c_idx = int(a_idx), int(b_idx), int(c_idx)
        progress = []
        if self._cur_step_qubits:
            self._steps += 1
            self._cur_step_qubits.clear()
        progress.append(line.format(self._steps, qasm, start_x + self._steps * gap_x + box_side / 2, self._qubit_y[a_name][a_idx], start_x + self._steps * gap_x + box_side / 2, self._qubit_y[b_name][b_idx], stroke_width))
        progress.append(line.format(self._steps, qasm, start_x + self._steps * gap_x + box_side / 2, self._qubit_y[b_name][b_idx], start_x + self._steps * gap_x + box_side / 2, self._qubit_y[c_name][c_idx], stroke_width))
        progress.append(line.format(self._steps, qasm, start_x + self._steps * gap_x + box_side / 2, self._qubit_y[a_name][a_idx], start_x + self._steps * gap_x + box_side / 2, self._qubit_y[c_name][c_idx], stroke_width))
        progress.append(dot.format(self._steps, qasm, start_x + self._steps * gap_x + box_side / 2, self._qubit_y[a_name][a_idx], dot_r, stroke_width))
        progress.append(dot.format(self._steps, qasm, start_x + self._steps * gap_x + box_side / 2, self._qubit_y[b_name][b_idx], dot_r, stroke_width))
        progress.append(circle.format(self._steps, qasm, start_x + self._steps * gap_x + box_side / 2, self._qubit_y[c_name][c_idx], circle_r, stroke_width))
        progress.append(line.format(self._steps, qasm, start_x + self._steps * gap_x + 2, self._qubit_y[c_name][c_idx], start_x + self._steps * gap_x + 2 * circle_r + 2, self._qubit_y[c_name][c_idx], stroke_width))
        progress.append(line.format(self._steps, qasm, start_x + self._steps * gap_x + circle_r + 2, self._qubit_y[c_name][c_idx] - circle_r, start_x + self._steps * gap_x + circle_r + 2, self._qubit_y[c_name][c_idx] + circle_r, stroke_width))

        self._steps += 1
        return progress

    def _generate_svg_for_unigate(self, qasm):
        gate, name, idx = re.split('\[|\s|\]', qasm)[:3]
        idx = int(idx)
        progress = []
        if name not in self._cur_step_qubits:
            self._cur_step_qubits[name] = [idx]
        else:
            if idx not in self._cur_step_qubits[name]:
                self._cur_step_qubits[name].append(idx)
            else:
                self._steps += 1
                self._cur_step_qubits.clear()
                self._cur_step_qubits[name] = [idx]
        if not gate == 'id':
            progress.append(rect.format(self._steps, qasm, start_x + self._steps * gap_x, self._qubit_y[name][idx] - box_side / 2, box_side, box_side, gate_color[gate]))
            progress.append(text_g.format(self._steps, qasm, start_x + self._steps * gap_x + box_side / 4 + gate_label_x_offset[gate], self._qubit_y[name][idx] + box_side / 4, gate[0].upper()))
        return progress

    def generate_svg(self):
        qubits = {}
        for l in self._qasm.split('\n'):
            l = l.strip()
            if l.find('qreg') == 0:
                _, name, num, _ = re.split('\[|\s|\]', l)
                qubits[name] = int(num)

        declarations = self._generate_svg_for_qubits_declarations(qubits)
        progress = []
        for l in self._qasm.split('\n'):
            l = l.strip()
            print l
            if l == '' or l.find('//') == 0 or l.find('include') == 0:
                # help with alignment for the amplitude amplification phase
                if l.find('// Amplitude amplification') >= 0 and self._cur_step_qubits:
                    # self._cur_step_qubits[config.VARIABLE_NAME] = [0]
                    self._steps += 1
                continue
            if l.find('qreg') == 0:
                continue
            elif l.find('creg') == 0:
                continue
            elif l.find('ccx') == 0:
                progress.extend(self._gen_svg_for_ccx(l))
            else:
                # if gate == 'h' or gate == 'x' or gate == 'measure':
                progress.extend(self._generate_svg_for_unigate(l))

        self._max_x = int([float(a.split('=')[1].replace('"', '')) for a in progress[-1].split() if a[0] == 'x'][0]) + box_side
        # return '<svg width="{}" height="{}">\n'.format(self._max_x + padding_right, self._max_y + padding_bottom) + '\n'.join([d.format(self._max_x) for d in declarations] + progress) + '\n</svg>'
        return '<svg class="generated-qasm-svg" width="100%" height="100%" viewBox="0 0 {} {}" preserveAspectRatio="xMaxYMax">\n'.format(self._max_x + padding_right, self._max_y + padding_bottom) + '\n'.join([d.format(self._max_x) for d in declarations] + progress) + '\n</svg>'
