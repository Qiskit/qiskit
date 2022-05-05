from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.visualization.exceptions import VisualizationError
try:
    import tkinter
    from tkinter import LEFT, END, DISABLED, NORMAL
except ImportError:
    raise MissingOptionalLibraryError(
        libname='Tkinter',
        name='quantum_glasses',
        pip_install='pip install tk'
    )
import warnings
from numpy import pi
from qiskit import QuantumCircuit
from qiskit.visualization import visualize_transition

# Ignore unnecessary warnings
warnings.simplefilter("ignore")

# Define the colors and fonts
background = '#2c94c8'
buttons = '#834558'
special_buttons = '#bc3454'
button_font = ('Arial', 18)
display_font = ('Arial', 32)

# Initialize the Quantum Circuit
def initialize_circuit():
    global circuit
    circuit = QuantumCircuit(1)

initialize_circuit()


theta = 0

# Define Functions
# Define functions for non-qiskit buttons
def display_gate(gate_input,display):
    """
    Adds a corresponding gate notation in the display to track the operations.
    If the number of operation reach ten, all gate buttons are disabled.
    """
    # Insert the defined gate
    display.insert(END,gate_input)


def clear(display):
    """
    Clears the display!
    Reintializes the Quantum Circuit for fresh calculation!
    """
    # clear the display
    display.delete(0, END)

    # reset the circuit to initial state |0> 
    initialize_circuit()


def about():
    """
    Displays the info about the project!
    Args: None
    Returns:
            Opens tkinter GUI and returns after the GUI is closed.
    """
    info = tkinter.Tk()
    info.title('About')
    info.geometry('650x470')
    info.resizable(0,0)

    text = tkinter.Text(info, height = 20, width = 20)

    # Create label
    label = tkinter.Label(info, text = "About Quantum Glasses:")
    label.config(font =("Arial", 14))
    

    text_to_display = """ 
    About: Visualization tool for Single Qubit Rotation on Bloch Sphere

    Info about the gate buttons and corresponding qiskit commands:

    X = flips the state of qubit -                                 circuit.x()
    Y = rotates the state vector about Y-axis -                    circuit.y()
    Z = flips the phase by PI radians -                            circuit.z()
    Rx = parameterized rotation about the X axis -                 circuit.rx()
    Ry = parameterized rotation about the Y axis.                  circuit.ry()
    Rz = parameterized rotation about the Z axis.                  circuit.rz()
    S = rotates the state vector about Z axis by PI/2 radians -    circuit.s()
    T = rotates the state vector about Z axis by PI/4 radians -    circuit.t()
    Sd = rotates the state vector about Z axis by -PI/2 radians -  circuit.sdg()
    Td = rotates the state vector about Z axis by -PI/4 radians -  circuit.tdg()
    H = creates the state of superposition -                       circuit.h()

    For Rx, Ry and Rz, 
    theta(rotation_angle) allowed range in the app is [-2*PI,2*PI]

    In case of a Visualization Error, the app closes automatically.
    This indicates that visualization of your circuit is not possible.

    """
    
    label.pack()
    text.pack(fill='both',expand=True)

    # Insert the text
    text.insert(END,text_to_display)

    # run
    info.mainloop()


def change_theta(num,window,circuit,key):
    """
    Changes the global variable theta and destroys the window
    """
    global theta
    theta = num * pi
    if key=='x':
        circuit.rx(theta,0)
        theta = 0
    elif key=='y':
        circuit.ry(theta,0)
        theta = 0
    else:
        circuit.rz(theta,0)
        theta = 0
    window.destroy()

def user_input(circuit,key):
    """
    Take the user input for rotation angle for parameterized 
    Rotation gates Rx, Ry, Rz.
    Args:
        circuit(QuantumCircuit): Qiskit single-qubit QuantumCircuit
        key(string): A single character string with values either "x" or "y"

    Returns:
            Opens tkinter GUI and returns after the GUI is closed.
    """

    # Initialize and define the properties of window
    get_input = tkinter.Tk()
    get_input.title('Get Theta')
    get_input.geometry('360x160')
    get_input.resizable(0,0)

    val1 = tkinter.Button(get_input,height=2,width=10,bg=buttons,font=("Arial",10),text='PI/4',command=lambda:change_theta(0.25,get_input,circuit,key))
    val1.grid(row=0, column=0)

    val2 = tkinter.Button(get_input,height=2,width=10,bg=buttons,font=("Arial",10),text='PI/2',command=lambda:change_theta(0.50,get_input,circuit,key))
    val2.grid(row=0, column=1)

    val3 = tkinter.Button(get_input,height=2,width=10,bg=buttons,font=("Arial",10),text='PI',command=lambda:change_theta(1.0,get_input,circuit,key))
    val3.grid(row=0, column=2)

    val4 = tkinter.Button(get_input,height=2,width=10,bg=buttons,font=("Arial",10),text='2*PI',command=lambda:change_theta(2.0,get_input,circuit,key))
    val4.grid(row=0, column=3,sticky='W')

    nval1 = tkinter.Button(get_input,height=2,width=10,bg=buttons,font=("Arial",10),text='-PI/4',command=lambda:change_theta(-0.25,get_input,circuit,key))
    nval1.grid(row=1, column=0)

    nval2 = tkinter.Button(get_input,height=2,width=10,bg=buttons,font=("Arial",10),text='-PI/2',command=lambda:change_theta(-0.50,get_input,circuit,key))
    nval2.grid(row=1, column=1)

    nval3 = tkinter.Button(get_input,height=2,width=10,bg=buttons,font=("Arial",10),text='-PI',command=lambda:change_theta(-1.0,get_input,circuit,key))
    nval3.grid(row=1, column=2)
    
    nval4 = tkinter.Button(get_input,height=2,width=10,bg=buttons,font=("Arial",10),text='-2*PI',command=lambda:change_theta(-2.0,get_input,circuit,key))
    nval4.grid(row=1, column=3,sticky='W')

    text_object = tkinter.Text(get_input, height = 20, width = 20,bg="light cyan")

    note = """
    GIVE THE VALUE FOR THETA
    The value has the range [-2*PI,2*PI]
    """

    text_object.grid(sticky='WE',columnspan=4)
    text_object.insert(END,note)


    get_input.mainloop()

# Define functions for qiskit-based buttons

## Define the function for visualize button
def visualize_circuit(circuit,window):
    """
    Visualizes the single qubit rotations corresponding to applied gates in a separate tkinter window.
    Handles any possible visualization error
    Args:
        circuit(QuantumCircuit): Qiskit single-qubit QuantumCircuit
        window(Tk): Tk class object tkinter window

    Returns:
            Opens tkinter GUI and returns after the GUI is closed.
    Raises:
           VisualizationError: Given gate(s) are not supported or there is not visualization possible.
    """
    try:
        visualize_transition(circuit=circuit)
    except VisualizationError as error:
        window.destroy()
        raise error

## The main function to initialize GUI
def quantum_glasses():
    """
    Provides a scientific calculator-like Tkinter GUI to the user for visualizing the single-qubit state transition over the Bloch Sphere
    with animations supported by the function qiskit.visualization.visualize_transition. The GUI can be only used inside a python script.
    Args: None

    Returns: returns after the GUI window is closed

    Raises:
           ModuleNotFoundError: must have Tkinter 
           PythonScriptNotFoundError: must be executed inside a python script

    Example:
    from qiskit.visualization.quantum_glasses import quantum_glasses
    quantum_glasses()
    """

    try:
        shell = get_ipython().__class__.__name__
        if shell=='ZMQInteractiveShell' or shell=='TerminalInteractiveShell':
            is_notebook = True
    except NameError:
        is_notebook=False

    if is_notebook:
        raise Exception("PythonScriptNotFound: must be executed inside a python script")
        
    # Define Window
    root = tkinter.Tk()
    root.title('Quantum Glasses')

    # set the icon
    root.iconbitmap(default='logo.ico')
    root.geometry('399x410')
    root.resizable(0,0) # Blocking the resizing feature

    # Define Layout
    # Define the Frames
    display_frame = tkinter.LabelFrame(root)
    button_frame = tkinter.LabelFrame(root,bg='black')
    display_frame.pack()
    button_frame.pack(fill='both',expand=True)

    # Define the Display Frame Layout
    display = tkinter.Entry(display_frame, width=120, font=display_font, bg=background, borderwidth=2, justify=LEFT)
    display.pack(padx=3,pady=4)

    # Define the Button Frame Layout

    # Define the first row of buttons
    x_gate = tkinter.Button(button_frame, font=button_font, bg=buttons, text='X',command=lambda:[display_gate('x',display),circuit.x(0)])
    y_gate = tkinter.Button(button_frame, font=button_font, bg=buttons, text='Y',command=lambda:[display_gate('y',display),circuit.y(0)])
    z_gate = tkinter.Button(button_frame, font=button_font, bg=buttons, text='Z',command=lambda:[display_gate('z',display),circuit.z(0)])
    x_gate.grid(row=0,column=0,ipadx=45, pady=1)
    y_gate.grid(row=0,column=1,ipadx=45, pady=1)
    z_gate.grid(row=0,column=2,ipadx=53, pady=1, sticky='E')

    
    # Define the second row of buttons
    Rx_gate = tkinter.Button(button_frame, font=button_font, bg=buttons, text='RX',command=lambda:[display_gate('Rx',display),user_input(circuit,'x')])
    Ry_gate = tkinter.Button(button_frame, font=button_font, bg=buttons, text='RY',command=lambda:[display_gate('Ry',display),user_input(circuit,'y')])
    Rz_gate = tkinter.Button(button_frame, font=button_font, bg=buttons, text='RZ',command=lambda:[display_gate('Rz',display),user_input(circuit,'z')])
    Rx_gate.grid(row=1,column=0,columnspan=1,sticky='WE', pady=1)
    Ry_gate.grid(row=1,column=1,columnspan=1,sticky='WE', pady=1)
    Rz_gate.grid(row=1,column=2,columnspan=1,sticky='WE', pady=1)

    # Define the third row of buttons
    s_gate = tkinter.Button(button_frame, font=button_font, bg=buttons, text='S',command=lambda:[display_gate('s',display),circuit.s(0)])
    sd_gate = tkinter.Button(button_frame, font=button_font, bg=buttons, text='SD',command=lambda:[display_gate('SD',display),circuit.sdg(0)])
    hadamard = tkinter.Button(button_frame, font=button_font, bg=buttons, text='H',command=lambda:[display_gate('H',display),circuit.h(0)])
    s_gate.grid(row=2,column=0,columnspan=1,sticky='WE', pady=1)
    sd_gate.grid(row=2,column=1,sticky='WE', pady=1)
    hadamard.grid(row=2, column=2, rowspan=2,sticky='WENS', pady=1)

    # Define the fifth row of buttons
    t_gate = tkinter.Button(button_frame, font=button_font, bg=buttons, text='T', command=lambda:[display_gate('t',display),circuit.t(0)])
    td_gate = tkinter.Button(button_frame, font=button_font, bg=buttons, text='TD',command=lambda:[display_gate('TD',display),circuit.tdg(0)])
    t_gate.grid(row=3,column=0,sticky='WE', pady=1)
    td_gate.grid(row=3,column=1,sticky='WE', pady=1)

    # Define the Quit and Visualize buttons
    quit = tkinter.Button(button_frame, font=button_font, bg=special_buttons, text='Quit',command=root.destroy)
    visualize = tkinter.Button(button_frame, font=button_font, bg=special_buttons, text='Visualize',command=lambda:visualize_circuit(circuit,root))
    quit.grid(row=4,column=0,columnspan=2,sticky='WE',ipadx=5, pady=1)
    visualize.grid(row=4,column=2,columnspan=1,sticky='WE',ipadx=8, pady=1)

    # Define the clear button
    clear_button = tkinter.Button(button_frame, font=button_font, bg=special_buttons, text='Clear',command=lambda:clear(display))
    clear_button.grid(row=5,column=0,columnspan=3,sticky='WE')

    # Define the about button
    about_button = tkinter.Button(button_frame, font=button_font, bg=special_buttons, text='About',command=about)
    about_button.grid(row=6,column=0,columnspan=3,sticky='WE')

    # Run the main loop
    root.mainloop()

