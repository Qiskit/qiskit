import matplotlib
import matplotlib.pyplot

import numpy as np
from matplotlib import colors as mcolors


def plot_quantum_circuit(gates,inits={},labels=[],plot_labels=True,**kwargs):
    """Use Matplotlib to plot a quantum circuit.
    gates     List of tuples for each gate in the quantum circuit.
              (name,target,control1,control2...). Targets and controls initially
              defined in terms of labels. 
    inits     Initialization list of gates, optional
    
    kwargs    Can override plot_parameters
    """
    plot_params = dict(scale = 1.0,fontsize = 14.0, linewidth = 1.0, 
                         control_radius = 0.05, not_radius = 0.15, 
                         swap_delta = 0.08, label_buffer = 0.0)
    plot_params.update(kwargs)
    scale = plot_params['scale']
    
    # Create labels from gates. This will become slow if there are a lot 
    #  of gates, in which case move to an ordered dictionary
    if not labels:
        labels = []
        for i,gate in enumerate_gates(gates):
            for label in gate[1:]:
                if label not in labels:
                    labels.append(label)
    
    nq = len(labels)
    ng = len(gates)
    wire_grid = np.arange(0.0, nq*scale, scale, dtype=float)
    gate_grid = np.arange(0.0, ng*scale, scale, dtype=float)
    
    fig,ax = setup_figure(nq,ng,gate_grid,wire_grid,plot_params)

    measured = measured_wires(gates,labels)
    draw_wires(ax,nq,gate_grid,wire_grid,plot_params,measured)
    
    if plot_labels: 
        draw_labels(ax,labels,inits,gate_grid,wire_grid,plot_params)

    draw_gates(ax,gates,labels,gate_grid,wire_grid,plot_params,measured)
    matplotlib.pyplot.show()
    return ax

def enumerate_gates(l,schedule=False):
    "Enumerate the gates in a way that can take l as either a list of gates or a schedule"
    if schedule:
        for i,gates in enumerate(l):
            for gate in gates:
                yield i,gate
    else:
        for i,gate in enumerate(l):
            yield i,gate
    return

def measured_wires(l,labels,schedule=False):
    "measured[i] = j means wire i is measured at step j"
    measured = {}
    for i,gate in enumerate_gates(l,schedule=schedule):
        name,target = gate[:2]
        j = get_flipped_index(target,labels)
        if name in ['MEASURE', 'measure', 'M', 'm']: #name.startswith('M')
            measured[j] = i
    return measured

def draw_gates(ax,l,labels,gate_grid,wire_grid,plot_params,measured={},schedule=False):
    for i,gate in enumerate_gates(l,schedule=schedule):
        # IBM Q format is different from this.
        # IBM Q puts the target at the last entry, while this puts the target at the first entry after the gate name.
        # we convert IBM Q format to this format.
        if len(gate) > 2:
            gate = list(gate)
            tmp = gate[1]
            gate[1] = gate[-1]
            gate[-1] = tmp

        draw_target(ax,i,gate,labels,gate_grid,wire_grid,plot_params)
        if len(gate) > 2: # Controlled
            draw_controls(ax,i,gate,labels,gate_grid,wire_grid,plot_params,measured)
    return

def draw_controls(ax,i,gate,labels,gate_grid,wire_grid,plot_params,measured={}):
    linewidth = plot_params['linewidth']
    scale = plot_params['scale']
    control_radius = plot_params['control_radius']


    name,target = gate[:2]
    controls = gate[2:]


    target_index = get_flipped_index(target,labels)
    control_indices = get_flipped_indices(controls,labels)
    gate_indices = control_indices + [target_index]
    min_wire = min(gate_indices)
    max_wire = max(gate_indices)
    line(ax,gate_grid[i],gate_grid[i],wire_grid[min_wire],wire_grid[max_wire],plot_params)
    ismeasured = False
    for index in control_indices:
        if measured.get(index,1000) < i: 
            ismeasured = True
    if ismeasured:
        dy = 0.04 # TODO: put in plot_params
        line(ax,gate_grid[i]+dy,gate_grid[i]+dy,wire_grid[min_wire],wire_grid[max_wire],plot_params)
        
    for ci in control_indices:
        x = gate_grid[i]
        y = wire_grid[ci]
        if name in ['SWAP', 'swap']:
            swapx(ax,x,y,plot_params)
        else:
            cdot(ax,x,y,plot_params)
    return

def draw_target(ax,i,gate,labels,gate_grid,wire_grid,plot_params):
    target_symbols = dict(CNOT='X',CPHASE='Z',NOP='',CX='X',CZ='Z')
    name,target = gate[:2]
    symbol = target_symbols.get(name,name) # override name with target_symbols
    x = gate_grid[i]
    target_index = get_flipped_index(target,labels)
    y = wire_grid[target_index]
    if not symbol: return
    if name in ['CNOT', 'cnot', 'TOFFOLI',  'toffoli', 'CX', 'cx']:
        oplus(ax,x,y,plot_params)
    elif name in ['CPHASE', 'cphase']:
        cdot(ax,x,y,plot_params)
    elif name in ['SWAP', 'swap']:
        swapx(ax,x,y,plot_params)
    else:
        text(ax,x,y,symbol,plot_params,box=True)
    return

def line(ax,x1,x2,y1,y2,plot_params):
    Line2D = matplotlib.lines.Line2D
    line = Line2D((x1,x2), (y1,y2),
        color='k',lw=plot_params['linewidth'])
    ax.add_line(line)

#manual is here: https://matplotlib.org/examples/color/named_colors.html
def text(ax,x,y,textstr,plot_params,box=False):
    linewidth = plot_params['linewidth']
    fontsize = plot_params['fontsize']
    if box:
        # determine the fill color, to be consistent with IBM Q
        if textstr in ['ID', 'id']:
            fillcolor = mcolors.cnames['gold']
        elif textstr in ['X', 'Y', 'Z', 'x', 'y', 'z']:
            fillcolor = mcolors.cnames['lime']
        elif textstr in ['H', 'S', 'SDG', 'h', 's', 'sdg']:
            fillcolor = mcolors.cnames['deepskyblue']
        elif textstr in ['T', 'TDG', 't', 'tdg']:
            fillcolor = mcolors.cnames['tomato']
        else:
            fillcolor = 'w'

        bbox = dict(ec='k',fc=fillcolor,fill=True,lw=linewidth)
    else:
        bbox=False
    ax.text(x,y,textstr,color='k',ha='center',va='center',bbox=bbox,size=fontsize)
    return

def oplus(ax,x,y,plot_params):
    Line2D = matplotlib.lines.Line2D
    Circle = matplotlib.patches.Circle
    not_radius = plot_params['not_radius']
    linewidth = plot_params['linewidth']
    c = Circle((x, y),not_radius,ec='k',
               fc=mcolors.cnames['deepskyblue'],fill=True,lw=linewidth)
    ax.add_patch(c)
    line(ax,x,x,y-not_radius,y+not_radius,plot_params)
    return

def cdot(ax,x,y,plot_params):
    Circle = matplotlib.patches.Circle
    control_radius = plot_params['control_radius']
    scale = plot_params['scale']
    linewidth = plot_params['linewidth']
    c = Circle((x, y),control_radius*scale,
        ec='k',fc=mcolors.cnames['deepskyblue'],fill=True,lw=linewidth)
    ax.add_patch(c)
    return

def swapx(ax,x,y,plot_params):
    d = plot_params['swap_delta']
    linewidth = plot_params['linewidth']
    line(ax,x-d,x+d,y-d,y+d,plot_params)
    line(ax,x-d,x+d,y+d,y-d,plot_params)
    return

def setup_figure(nq,ng,gate_grid,wire_grid,plot_params):
    scale = plot_params['scale']
    fig = matplotlib.pyplot.figure(
        figsize=(ng*scale, nq*scale),
        facecolor='w',
        edgecolor='w'
    )
    ax = fig.add_subplot(1, 1, 1,frameon=True)
    ax.set_axis_off()
    offset = 0.5*scale
    ax.set_xlim(gate_grid[0] - offset, gate_grid[-1] + offset)
    ax.set_ylim(wire_grid[0] - offset, wire_grid[-1] + offset)
    ax.set_aspect('equal')
    return fig,ax

def draw_wires(ax,nq,gate_grid,wire_grid,plot_params,measured={}):
    scale = plot_params['scale']
    linewidth = plot_params['linewidth']
    xdata = (gate_grid[0] - scale, gate_grid[-1] + scale)
    for i in range(nq):
        line(ax,gate_grid[0]-scale,gate_grid[-1]+scale,wire_grid[i],wire_grid[i],plot_params)
        
    # Add the doubling for measured wires:
    dy=0.04 # TODO: add to plot_params
    for i in measured:
        j = measured[i]
        line(ax,gate_grid[j],gate_grid[-1]+scale,wire_grid[i]+dy,wire_grid[i]+dy,plot_params)
    return

def draw_labels(ax,labels,inits,gate_grid,wire_grid,plot_params):
    scale = plot_params['scale']
    label_buffer = plot_params['label_buffer']
    fontsize = plot_params['fontsize']
    nq = len(labels)
    xdata = (gate_grid[0] - scale, gate_grid[-1] + scale)
    for i in range(nq):
        j = get_flipped_index(labels[i],labels)
        text(ax,xdata[0]-label_buffer,wire_grid[j],render_label(labels[i],inits),plot_params)
    return

def get_flipped_index(target,labels):
    """Get qubit labels from the rest of the line,and return indices

    get_flipped_index('q0', ['q0', 'q1'])
    1
    get_flipped_index('q1', ['q0', 'q1'])
    0
    """
    nq = len(labels)
    i = labels.index(target)
    return nq-i-1

def get_flipped_indices(targets,labels): return [get_flipped_index(t,labels) for t in targets]

def render_label(label, inits={}):
    """Slightly more flexible way to render labels.

    render_label('q0')
    '$|q0\\\\rangle$'
    render_label('q0', {'q0':'0'})
    '$|0\\\\rangle$'
    """
    if label in inits:
        s = inits[label]
        if s is None:
            return ''
        else:
            return r'$|%s\rangle$' % inits[label]
    return r'$|%s\rangle$' % label


def plot_quantum_schedule(schedule, inits={},labels=[],plot_labels=True,**kwargs):
    """Use Matplotlib to plot a quantum circuit.
    schedule  List of time steps, each containing a sequence of gates during that step.
              Each gate is a tuple containing (name,target,control1,control2...).
              Targets and controls initially defined in terms of labels.
    inits     Initialization list of gates, optional

    kwargs    Can override plot_parameters
    """
    plot_params = dict(scale = 1.0,fontsize = 14.0, linewidth = 1.0,
                         control_radius = 0.05, not_radius = 0.15,
                         swap_delta = 0.08, label_buffer = 0.0)
    plot_params.update(kwargs)
    scale = plot_params['scale']

    # Create labels from gates. This will become slow if there are a lot
    #  of gates, in which case move to an ordered dictionary
    if not labels:
        labels = []
        for i,gate in enumerate_gates(schedule,schedule=True):
            for label in gate[1:]:
                if label not in labels:
                    labels.append(label)

    nq = len(labels)
    nt = len(schedule)
    wire_grid = np.arange(0.0, nq*scale, scale, dtype=float)
    gate_grid = np.arange(0.0, nt*scale, scale, dtype=float)

    fig,ax = setup_figure(nq,nt,gate_grid,wire_grid,plot_params)

    measured = measured_wires(schedule,labels,schedule=True)
    draw_wires(ax,nq,gate_grid,wire_grid,plot_params,measured)

    if plot_labels:
        draw_labels(ax,labels,inits,gate_grid,wire_grid,plot_params)

    draw_gates(ax,schedule,labels,gate_grid,wire_grid,plot_params,measured,schedule=True)
    matplotlib.pyplot.show()


    return ax


def save_plot(png_path, schedule, inits={},labels=[],plot_labels=True,**kwargs):
    """Use Matplotlib to plot a quantum circuit.
    schedule  List of time steps, each containing a sequence of gates during that step.
              Each gate is a tuple containing (name,target,control1,control2...).
              Targets and controls initially defined in terms of labels.
    inits     Initialization list of gates, optional

    kwargs    Can override plot_parameters
    """
    plot_params = dict(scale = 1.0,fontsize = 14.0, linewidth = 1.0,
                         control_radius = 0.05, not_radius = 0.15,
                         swap_delta = 0.08, label_buffer = 0.0)
    plot_params.update(kwargs)
    scale = plot_params['scale']

    # Create labels from gates. This will become slow if there are a lot
    #  of gates, in which case move to an ordered dictionary
    if not labels:
        labels = []
        for i,gate in enumerate_gates(schedule,schedule=True):
            for label in gate[1:]:
                if label not in labels:
                    labels.append(label)

    nq = len(labels)
    nt = len(schedule)
    wire_grid = np.arange(0.0, nq*scale, scale, dtype=float)
    gate_grid = np.arange(0.0, nt*scale, scale, dtype=float)

    fig,ax = setup_figure(nq,nt,gate_grid,wire_grid,plot_params)

    measured = measured_wires(schedule,labels,schedule=True)
    draw_wires(ax,nq,gate_grid,wire_grid,plot_params,measured)

    if plot_labels:
        draw_labels(ax,labels,inits,gate_grid,wire_grid,plot_params)

    draw_gates(ax,schedule,labels,gate_grid,wire_grid,plot_params,measured,schedule=True)
    # matplotlib.pyplot.show()

    fig.savefig(png_path)   # save the figure to file
    matplotlib.pyplot.close(fig)

    return ax


# H,X,Y,Z,S,T,M = 'HXYZSTM'
# CNOT,CPHASE,CZ,CX,TOFFOLI,SWAP,NOP = 'CNOT','CPHASE','CZ','CX','TOFFOLI','SWAP','NOP'
# qa,qb,qc,qd,q0,q1,q2,q3 = 'q_a','q_b','q_c','q_d','q_0','q_1','q_2','q_3'

# example:
# plot_quantum_circuit([('H','j_0'),('S','j_0','j_1'),('T','j_0','j_2'),('H','j_1'), ('S','j_1','j_2'),('H','j_2'),('CNOT','j_0','j_2')])

# example:
#plot_quantum_schedule([[('H','q0')], [('CNOT','q1','q0')]])
