# Copyright 2017 James R. Wootton. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append("../../../")
from IBMQuantumExperience import IBMQuantumExperience
from qiskit import QuantumProgram
import Qconfig

import getpass, random, numpy, math, time

def setText ():
    # this function outputs a list of strings, to be used as the wait text
    
    wait_text = []
    
    wait_text.append("* Your ships and bombs have now been turned into a quantum program.")
    wait_text.append("* That program is in a queue to be processed by a real quantum device.")
    wait_text.append("* It is the ibmqx2, made by IBM.")
    wait_text.append("* You can also play with ibmqx2 using the IBM Quantum Experience. Check it out!")
    wait_text.append("* This device is the world’s only publicly accessible quantum computer (in July 2017).")
    wait_text.append("* So we could be in the queue for a minute or two.")
    wait_text.append("* I’ll use that time to tell you about quantum computers, and how this program works.")
    wait_text.append("* You can also find all this stuff at\nhttps://medium.com/@decodoku/quantum-computation-in-84-short-lines-d9c7c74be0d0")
    wait_text.append("* Firstly, let’s make sure we all know what a normal computer is.")
    wait_text.append("* Information is stored using lots of bits. Each can be either 0 or 1.")
    wait_text.append("* These are manipulated using logic gates, built from transistors.")
    wait_text.append("* The simplest logic gate is the NOT. It turns a 0 to a 1, and vice-versa.")
    wait_text.append("* Another fun logic gate is AND. It takes two bits and looks whether both are 1.")
    wait_text.append("* There are also others, like XOR, but they can all be built from NOTs and ANDs.")
    wait_text.append("* In fact, any program can be compiled into a whole bunch of NOTs and ANDs.")
    wait_text.append("* They are the building blocks of computing.")
    wait_text.append("* But some programs need LOADS of these blocks.")
    wait_text.append("* This means they’d need a huge computer, and it would need to run for ages.")
    wait_text.append("* If a program could only run with a planet sized supercomputer…")
    wait_text.append("* …which needs to run for the age of the universe…")
    wait_text.append("* …running it would be practically impossible.")
    wait_text.append("* Unless we find better building blocks!")
    wait_text.append("* That’s what quantum computers promise to do.")
    wait_text.append("* These are based on quantum bits, or ‘qubits’.")
    wait_text.append("* A qubit can either be 0 or 1. But it can also be a quantum superposition of the two.")
    wait_text.append("* We can think of these as states that exist partway between 0 and 1.")
    wait_text.append("* There are infinitely many kinds of these superpositions.")
    wait_text.append("* They’ll all act a bit differently when we apply gates to them.")
    wait_text.append("* At the end of the program, we look at whether each qubit is 0 or 1.")
    wait_text.append("* Since these are the only two options, superpositions will have to decide one way or the other.")
    wait_text.append("* The choice will always be random, but some superpositions are biased towards 0…")
    wait_text.append("* …and some are biased towards 1.")
    wait_text.append("* The existence of the superpositions allow us to do many more gates.")
    wait_text.append("* We could do half a NOT, for example.")
    wait_text.append("* Instead of going all the way from 0 to 1, we would park in a superposition halfway.")
    wait_text.append("* There’s also different routes we could take through the superpositions between 0 and 1.")
    wait_text.append("* Each gives us a different kind of NOT.")
    wait_text.append("* The AND will also get more interesting when the qubits it compares are in superpositions.")
    wait_text.append("* Any fraction of an AND is also possible.")
    wait_text.append("* So are types of gate that are completely different to those on normal computers.")
    wait_text.append("* The building blocks of computation become a lot more malleable with qubits.")
    wait_text.append("* That means we can build software in completely new ways.")
    wait_text.append("* Problems that are currently practically impossible to solve will become easy.")
    wait_text.append("* All we need is to build a powerful enough quantum computer.")
    wait_text.append("* The device we are using here is a five qubit processor.")
    wait_text.append("* We’ll need a few more than that to solve big problems.")
    wait_text.append("* But for now, we can play Battleships!")
    wait_text.append("* In this game, each of the five positions corresponds to a qubit on the processor.")
    wait_text.append("* For positions that hold a ship, we use 0 to mean that it is intact and 1 for destroyed.")
    wait_text.append("* For the weakest ships, the bomb is implemented with a NOT gate.")
    wait_text.append("* One of these rotates all the way from 0 to 1, sinking the ship.")
    wait_text.append("* We use a particular kind of quantum NOT called an X.")
    wait_text.append("* For the ships need two bombs to sink, each bomb is done with half an X.")
    wait_text.append("* One of these rotates the ship to a superposition half way between 0 and 1.")
    wait_text.append("* If the program ends at this point, the qubit will randomly choose between 0 and 1.")
    wait_text.append("* Both options will be equally likely.")
    wait_text.append("* By running the program many times, we’ll find half the results for this ship are 0…")
    wait_text.append("* …and the other half are 1. We say that this has 50% damage.")
    wait_text.append("* But if the qubit gets a second half X before the program ends…")
    wait_text.append("* …it will carry on all the way to 1, and the ship is destroyed!")
    wait_text.append("* If it then gets another half X, it will start the journey back to 0.")
    wait_text.append("* Probably best not to resurrect ships like this if you want to win!")
    wait_text.append("* For the ship that needs three bombs, we use a third of an X.")
    wait_text.append("* So with a single quantum bit, and a type of quantum NOT gate…")
    wait_text.append("* …we can simulate a ship that needs three bombs to be destroyed.")
    wait_text.append("* If we used a normal computer, a single bit and some NOT gates wouldn’t be enough.")
    wait_text.append("* We’d need at least two bits for a such a ship, with something like 00 for intact…")
    wait_text.append("* …01 for partly damaged, 10 for mostly destroyed, and 11 for sunk.")
    wait_text.append("* Implementing the bombs would not just need NOT gates…")
    wait_text.append("* …since what you need to do to one bit depends on what the other is doing.")
    wait_text.append("* This would require XOR gates as well as NOTs.")
    wait_text.append("* An AND would also be needed to check if the ship is sunk.")
    wait_text.append("* It’s a lot more complex than just the qubit and partial NOTs that we use")
    wait_text.append("* This is a small example of how quantum programs could be more efficient.")
    wait_text.append("* Though don't take it too seriously. It does cheat a bit.")
    wait_text.append("* The effect is far more potent for other examples, like Shor's algorithm.")
    wait_text.append("* But they aren’t so fun!")
    wait_text.append("* Another thing to tell you about is noise.")
    wait_text.append("* Quantum computers are still a bit noisy.")
    wait_text.append("* So the damages you see in the game might not be quite what you expect.")
    wait_text.append("* You can interpret this as the effects of a stormy sea if you want.")
    wait_text.append("* I think that ends my little lecture.")
    wait_text.append("* If you want to know more about how to play with quantum programming…")
    wait_text.append("* …check us out at medium.com/@decodoku.")
    wait_text.append("* That’s all I have to tell you.")
    wait_text.append("* This message will repeat in 5…")
    wait_text.append("* …4…")
    wait_text.append("* …3…")
    wait_text.append("* …2…")
    wait_text.append("* …1…")
    
    return wait_text


def runGame():
    # this function runs the game!
    
    print("\n\n\n\n\n\n\n\n")
    print("            ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗   ██╗███╗   ███╗            ")
    print("           ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║   ██║████╗ ████║            ")
    print("           ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║            ")
    print("           ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║            ")
    print("           ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║            ")
    print("            ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝            ")
    print("")
    print("   ██████╗  █████╗ ████████╗████████╗██╗     ███████╗███████╗██╗  ██╗██╗██████╗ ███████╗")
    print("   ██╔══██╗██╔══██╗╚══██╔══╝╚══██╔══╝██║     ██╔════╝██╔════╝██║  ██║██║██╔══██╗██╔════╝")
    print("   ██████╔╝███████║   ██║      ██║   ██║     █████╗  ███████╗███████║██║██████╔╝███████╗")
    print("   ██╔══██╗██╔══██║   ██║      ██║   ██║     ██╔══╝  ╚════██║██╔══██║██║██╔═══╝ ╚════██║")
    print("   ██████╔╝██║  ██║   ██║      ██║   ███████╗███████╗███████║██║  ██║██║██║     ███████║")
    print("   ╚═════╝ ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚══════╝")
    print("")
    print("                 ___         ___                    _       _         ")
    print("                | _ ) _  _  |   \  ___  __  ___  __| | ___ | |__ _  _ ")
    print("                | _ \| || | | |) |/ -_)/ _|/ _ \/ _` |/ _ \| / /| || |")
    print("                |___/ \_, | |___/ \___|\__|\___/\__,_|\___/|_\_\ \_,_|")
    print("                      |__/                                            ")
    print("")
    print("")
    print("")
    print("         Learn how to make your own game for a quantum computer at decodoku.com")
    print("")
    print("")
    print("")
    print("   =====================================================================================")

    print("")
    print("")
    input("> Press Enter to continue...")
    print("")
    print("")
    print("                                     HOW TO PLAY")
    print("")
    
    
    print("Quantum Battleships is a game of two players.")
    print("")   
    print("Both players have a grid with 5 positions, labelled 0 to 4.")
    print("")
    print(" 4       0")
    print(" |\     /|")
    print(" | \   / |")
    print(" |  \ /  |")
    print(" |   2   |")
    print(" |  / \  |")
    print(" | /   \ |")
    print(" |/     \|")
    print(" 3       1")
    print("")
    input("> Press Enter to continue...")    
    print("")
    print("")
    print("")
    print("The players start by placing 3 ships on their own grid.")
    print("")
    print("Each ship takes up a single position")
    print("")
    print("The first ship placed by each player is the weakest. It can be destroyed by a single bomb.")
    print("")
    print("Two bombs are needed to destroy the second ship, and three bombs for the third.")
    print("")      
    input("> Press Enter to continue...")
    print("")
    print("")
    print("")
    print("The players take it in turns to bomb a position on their opponent's grid.")
    print("")
    print("If a ship is hit, the amount of damage is shown.")
    print("")
    print("Once a player destroys all their opponent's ships, the game is over.")
    print("")    
    input("> Press Enter to start setting up the game...")
    
    # we'll start by checking the device status
    # it seems that this needs a program to be set up, which is a bit of a pain
    # probably there's a better way, but this is how we are doing it
    print("")
    QPS_SPECS = {
    "name": "program-name",
    "circuits": [{
        "name": "circuitName",
        "quantum_registers": [{
            "name": "qname",
            "size": 3}],
        "classical_registers": [{
            "name": "cname",
            "size": 3}]
        }]
    }
    temp_program = QuantumProgram(specs=QPS_SPECS)
    apiconnection = temp_program.set_api(Qconfig.APItoken, Qconfig.config["url"])
    deviceStatus = temp_program.get_device_status('ibmqx2')
    
    print("")
    print("")
    print("   =====================================================================================")
    
    print("")
    print("")
    print("                                  CHOOSE YOUR DEVICE")
    print("")

    
    # note that device should be 'ibmqx_qasm_simulator', 'ibmqx2' or 'local_qasm_simulator'
    if (deviceStatus['available']):
        d = input("> Do you want to play on the real quantum computer? (y/n)\n")
        if ((d=="y")|(d=="Y")):
            device = 'ibmqx2'
        else:
            device = 'local_qasm_simulator'
            print("\nYou've chosen not to use a real quantum device")
    else:
        device = 'local_qasm_simulator'
        print("The real quantum computer is currently unavailable")
        
    if (device=='ibmqx2'):
        print("\nGreat choice!")
        print("The device is in IBM's Thomas J. Watson Research Center.")
        print("We'll send jobs to it and receive results via the internet.\n")
    else:
        
        print("Instead we'll get this computer to simulate one.")
        print("We can do this only because current quantum devices are small.")
        print("Soon we'll build ones that even a planet sized supercomputer could not simulate!\n")
    
    # while we are at it, let's set the number of shots
    shots = 1024

    # and also populate the wait text
    wait_text = setText()
    
    inputString = input("> Press Enter to start placing ships, or press R to restart the game...\n")
    # if we chose to restart, we do that
    if ((inputString=="r")|(inputString=="R")):
        return
    print("")
    print("")
    print("                                   PLACE YOUR SHIPS")
    print("")
    
    # The variable ship[X][Y] will hold the position of the Yth ship of player X+1
    shipPos = [ [-1]*3 for _ in range(2)] # all values are initialized to the impossible position -1|

    # loop over both players and all three ships for each
    for player in [0,1]:
        
        print("")
        print("PLAYER " + str(player+1))
        print("")
        
        # otherwise we ask for ship positions
        for ship in [0,1,2]:

            # ask for a position for each ship, and keep asking until a valid answer is given
            choosing = True
            while (choosing):

                # get player input
                position = getpass.getpass("> Choose a position for ship " + str(ship+1) + " (0, 1, 2, 3 or 4)\n" )

                # see if the valid input and ask for another if not
                if position.isdigit(): # valid answers  have to be integers
                    position = int(position)
                    if (position in [0,1,2,3,4]) and (not position in shipPos[player]): # they need to be between 0 and 5, and not used for another ship of the same player
                        shipPos[player][ship] = position
                        choosing = False
                        print ("")
                    elif position in shipPos[player]:
                        print("\nYou already have a ship there. Try again.\n")
                    else:
                        print("\nThat's not a valid position. Try again.\n")
                else:
                    print("\nThat's not a valid position. Try again.\n")


    # the game variable will be set to False once the game is over
    game = True

    # the variable bombs[X][Y] will hold the number of times position Y has been bombed by player X+1
    bomb = [ [0]*5 for _ in range(2)] # all values are initialized to zero

    # the variable grid[player] will hold the results for the grid of each player
    grid = [{},{}]

    round  = 0 # counter for rounds
    text_start = 0 # counter for wait_text
    while (game):
        
        round += 1
        
        inputString = input("> Press Enter to start round " + str(round) + ", or press R to restart the game...\n")
        # if we chose to restart, we do that
        if ((inputString=="r")|(inputString=="R")):
            return  
        print("")
        print("")
        print("                             ROUND " + str(round) + ": CHOOSE WHERE TO BOMB")
        print("")
        
        # ask both players where they want to bomb
        for player in range(2):

            print("")
            print("PLAYER " + str(player+1))
            print("")

            # keep asking until a valid answer is given
            choosing = True
            while (choosing):

                # get player input
                position = input("> Choose a position to bomb (0, 1, 2, 3 or 4)\n")

                # see if this is a valid input. ask for another if not
                if position.isdigit(): # valid answers  have to be integers
                    position = int(position)
                    if position in range(5): # they need to be between 0 and 5, and not used for another ship of the same player
                        bomb[player][position] = bomb[player][position] + 1
                        choosing = False
                        print ("\n")
                    else:
                        print("\nThat's not a valid position. Try again.\n")
                else:
                    print("\nThat's not a valid position. Try again.\n")

                    
        print("")
        print("")
        print("                                LET THE BOMBING BEGIN!")
        print("")

        
        if device=='ibmqx2':
            message = "\n> Press Enter to get the quantum computer to calculate what happens when the bombs hit,\nor press R to restart the game...\n"
        else:
            message = "\n> Press Enter to simulate what happens when the bombs hit, or press R to restart the game...\n"
        inputString = input(message)
        # if we chose to restart, we do that
        if ((inputString=="r")|(inputString=="R")):
            return
                    
                    
        # now we create and run the quantum programs that implement this on the grid for each player
        for player in range(2):

            # create a dictionary with the specifications of the program
            # we'll use all 5 qubits and bits, to avoid bugs on IBM's end
            Q_SPECS = {
            "circuits": [{
                "name": "gridScript",
                "quantum_registers": [{
                    "name": "q",
                    "size": 5
                }],
                "classical_registers": [{
                    "name": "c",
                    "size": 5
                }]}],
            }
            
            if device=='ibmqx2':
                print("\nWe'll now get the quantum computer to see what happens to Player " + str(player+1) + "'s ships.\n")
            else:
                print("\nWe'll now simulate what happens to Player " + str(player+1) + "'s ships.\n")
                
            # create the program with these specs
            Q_program = QuantumProgram(specs=Q_SPECS)

            # get the circuit by name
            gridScript = Q_program.get_circuit("gridScript")
            # get the quantum register by name
            q = Q_program.get_quantum_registers("q")
            # get the classical register by name
            c = Q_program.get_classical_registers("c")

            # add the bombs (of the opposing player)
            for position in range(5):
                # add as many bombs as have been placed at this position
                for n in range( bomb[(player+1)%2][position] ):
                    # the effectiveness of the bomb
                    # (which means the quantum operation we apply)
                    # depends on which ship it is
                    for ship in [0,1,2]:
                        if ( position == shipPos[player][ship] ):
                            frac = 1/(ship+1)
                            # add this fraction of a NOT to the QASM
                            gridScript.u3(frac * math.pi, 0.0, 0.0, q[position])

            #finally, measure them
            for position in range(5):
                gridScript.measure(q[position], c[position])

            # to see what the quantum computer is asked to do, we can print the QASM file
            # this lines is typically commented out
            #print( Q_program.get_qasm("gridScript") )

            # set the APIToken and API url
            Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"])

            # run the job until actual results are given
            dataNeeded = True
            while dataNeeded:

                try:
                #if(True):    
                    
                    # compile and run the QASM
                    runStatus = Q_program.execute(["gridScript"], device, shots, wait=2, timeout=600, wait_text = wait_text, text_start = text_start)        
                    
                    if 'waits' in runStatus.keys():
                        text_start += runStatus['waits']

                    # extract data
                    grid[player] = Q_program.get_counts("gridScript") 

                    if ('status' not in grid[player].keys()):
                        dataNeeded = False
                    else:
                        input("\n> This attempt at running on the quantum computer has failed\n"+
                              "Press Enter to try again, or R to restart the game.")
                        if ((inputString=="r")|(inputString=="R")):
                            return
                        
                #else:       
                except:
                    print("\n")
                    print("Something went wrong.\n")
                    time.sleep(2.0)
                    print("The quantum computer could be in maintenance.\n")
                    time.sleep(2.0)
                    print("Or your internet connection could be down.\n")
                    time.sleep(2.0)
                    print("Or some other gremlin.\n")
                    time.sleep(2.0)
                    print("Let's restart the game and try again.\n\n")
                    time.sleep(2.0)
                    return

            time.sleep(2.0)


        # we can check up on the data if we want
        # these lines are typically commented out
        #print( grid[0] )
        #print( grid[1] )
        

        # if one of the runs failed, tell the players and start the round again
        if ( ( 'Error' in grid[0].values() ) or ( 'Error' in grid[1].values() ) ):

            print("\nThe process timed out. Try this round again.\n")

        else:

            # look at the damage on all qubits (we'll even do ones with no ships)
            damage = [ [0]*5 for _ in range(2)] # this will hold the prob of a 1 for each qubit for each player

            # for this we loop over all 5 bit strings for each player
            for player in range(2):
                for bitString in grid[player].keys():
                    # and then over all positions
                    for position in range(5):
                        # if the string has a 1 at that position, we add a contribution to the damage
                        # remember that the bit for position 0 is the rightmost one, and so at bitString[4]
                        if (bitString[4-position]=="1"):
                            damage[player][position] += grid[player][bitString]/shots          

            # give results to players
            for player in [0,1]:

                input("\n> Press Enter to see the results for Player " + str(player+1) + "'s ships...")
                
                print("")
                print("")
                print("PLAYER " + str(player+1))
                print("")
                print("")       
                
                # report damage for qubits that are ships, and which have significant damange
                # ideally this would be non-zero damage, but noise means that can happen for ships that haven't been hit
                # so we choose 10% as the threshold
                display = [" ?  "]*5
                # loop over all qubits that are ships
                for position in shipPos[player]:
                    # if the damage is high enough, display the damage
                    if ( damage[player][position] > 0.1 ):
                        if (damage[player][position]>0.9):
                             display[position] = "100%"
                        else:
                            display[position] = str(int( 100*damage[player][position] )) + "% "

                print("Here is the percentage damage for ships that have been bombed.\n")
                print(display[ 4 ] + "    " + display[ 0 ])
                print(" |\     /|")
                print(" | \   / |")
                print(" |  \ /  |")
                print(" |  " + display[ 2 ] + " |")
                print(" |  / \  |")
                print(" | /   \ |")
                print(" |/     \|")
                print(display[ 3 ] + "    " + display[ 1 ])
                print("\n")
                print("Only ships with 100% damage have been destroyed\n")

                # if a player has all their ships destroyed, the game is over
                # ideally this would mean 100% damage, but we go for 95% because of noise again
                if (damage[player][ shipPos[player][0] ]>.95) and (damage[player][ shipPos[player][1] ]>.95) and (damage[player][ shipPos[player][2] ]>.95):
                    print ("         All Player " + str(player+1) + "'s ships have been destroyed!")
                    print ("")
                    game = False

            if (game is False):
                print("                                      GAME OVER")
                print ("")
                print ("")
                input("> Press Enter for a new game...\n")

                
while (True):
    
    runGame()