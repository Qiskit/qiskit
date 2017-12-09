# _basic_gates_string_IBM_advanced="id,x,y,z,h,s,sdg,cx,t,tdg,u1,u2,u3,cy,cz,ccx,cu1,cu3"
import random
import math
# by default, we do not support u gates since they are complex and does not have concrete meanings
unaryOPs = ['x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg'] # id
binaryOPs = ['cx', 'cy', 'cz'] # cx
unary_binary_ops = None
ternaryOPs = ['ccx'] #ccx

allOPs = ['x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'cx', 'cy', 'cz', 'ccx']

def ugate_wrap(ugatename):
    if ugatename == 'u1' or ugatename == 'cu1':
        p1 = random.uniform(0,math.pi)
        return ugatename+"("+str(p1) + ")"
    elif ugatename == 'u2':
        p1 = random.uniform(0,math.pi)
        p2 = random.uniform(0,math.pi)
        return ugatename+"("+str(p1) + "," + str(p2)+ ")"
    elif ugatename == 'u3' or ugatename == 'cu3':
        p1 = random.uniform(0,math.pi)
        p2 = random.uniform(0,math.pi)
        p3 = random.uniform(0,math.pi)
        return ugatename+"("+str(p1) + "," + str(p2)+ "," + str(p3)+ ")"
    else:
        return ugatename


def getDifferentGate(gateSignature):
    if "(" in gateSignature: # u1(p1...)
        gatename = gateSignature[0:gateSignature.index("(")]
    else:
        gatename = gateSignature

    if gatename in unaryOPs:
        unaryOPs.remove(gatename)
        newgatename = random.choice(unaryOPs)
        unaryOPs.append(gatename)
        return ugate_wrap(newgatename)
    elif gatename in binaryOPs:
        binaryOPs.remove(gatename)
        newgatename = random.choice(binaryOPs)
        binaryOPs.append(gatename)
        return ugate_wrap(newgatename)
    elif gatename in ternaryOPs:
        if len(ternaryOPs) > 1: # what if we only have one option: ccx?
            ternaryOPs.remove(gatename)
            newgatename = random.choice(ternaryOPs)
            ternaryOPs.append(gatename)
            return ugate_wrap(newgatename)
        else:
            return gatename # same old


# formatted:
#['h', 'q[1]']
# #['cx', 'q[1],q[2]']
def replaceGate(gatename, datasec):
    newgatename = getDifferentGate(gatename)
    return newgatename + " " + datasec



def getDifferentDatasection(datasec, bitDomain):
    datasecList = datasec.split(",")
    size = len(datasecList)

    if size == 1:
        if len(bitDomain) == 1:
            return datasec # no other choice
        else:
            bitDomain.remove(datasec)
            newdatasec = random.choice(bitDomain)
            bitDomain.append(datasec)
            return newdatasec
    else:
        newList = random.sample(bitDomain, size)
        while newList == datasecList: # redo it if duplication
            # we treat "different order" equivalently important as "different content"
            newList = random.sample(bitDomain, size)

        return ",".join(newList)


# formatted: ['cx', 'q[1],q[2]']
#replacements: ['q[3]', 'q[1]', 'q[2]']
def replaceBits(gatename, datasec, bitDomain):
    # pure: Note replacements may be all qubits or all cbits.
    # distint: after the replacement, the stmt should not contain two occurrences of same qubit/cbit
    newdatasec = getDifferentDatasection(datasec, bitDomain)
    return gatename + " " + newdatasec


def buildRandomInstruction(bitDomain):
    global unary_binary_ops
    if unary_binary_ops == None:
        unary_binary_ops = unaryOPs + binaryOPs # lazy initialization, wait for the main which may modify unaryOPs/binaryOPs


    bitDomainLen = len(bitDomain)
    if  bitDomainLen >= 3:
        randomgate = random.choice(allOPs)
    elif bitDomainLen == 2:
        randomgate = random.choice(unary_binary_ops)
    elif bitDomainLen == 1:
        randomgate = random.choice(unaryOPs) # can only pick unarygate
    else:
        raise SystemError()


    if randomgate in unaryOPs:
        size = 1
    elif randomgate in binaryOPs:
        size = 2
    elif randomgate in ternaryOPs:
        size = 3
    else:
        raise SystemError()

    try:
        newList = random.sample(bitDomain, size)
    except:
        newList = random.sample(bitDomain, size)


    randomdata = ",".join(newList)
    return ugate_wrap(randomgate) + " " + randomdata



def replaceInstruction(gatename, datasec, bitDomain):
    instr = buildRandomInstruction(bitDomain)
    oldinstr = gatename + " " + datasec
    while instr == oldinstr:
        instr = buildRandomInstruction(bitDomain)
    return instr



def mutate(orig_file, bitDomain, newFilePath, mutationOptionArg):
    _data = open(orig_file).readlines()
    lines = []
    for physicalLine in _data:
        if physicalLine.lstrip().startswith("//"):
            continue
        logicLines = physicalLine.split(";")
        for logicLine in logicLines:
            logicLine = logicLine.strip()
            if not logicLine or logicLine == '\n':
                continue
            lines.append(logicLine)


    header = []
    footer = []
    body = []
    for line in lines: # every line should be ended with a separator
        #print line

        if line.startswith("include ") or line.startswith("qreg ") or line.startswith("creg "):
            header.append(line)
        elif line.startswith("measure "): #TODO shall we also mutate this? currently no
            footer.append(line)
        else:
            body.append(line)

    # six mutations: 1 remove 2 insert 3 swap 4 replace gate 5 replace data 6 replace both
    if mutationOptionArg != None:
        mutationOption = mutationOptionArg
    else:
        mutationOption = random.randint(1,6) # including both ends

    bodyLength = len(body)
    if bodyLength == 1:
        mutationOption = 2  # force this, because your body is too short

    if mutationOption == 1:
        select = random.randint(0, bodyLength-1)
        del body[select]
    elif mutationOption == 2:
        injectPoint = random.randint(0, bodyLength)
        newline = buildRandomInstruction(bitDomain)
        body.insert(injectPoint, newline) # specially, insert at len(body) is same as appending to the end of list
    elif mutationOption == 3:
        selectedTwoIndexs = random.sample(range(bodyLength), 2) # range(5) return [0,1,2,3,4], not including 5
        index1 = selectedTwoIndexs[0]
        index2 = selectedTwoIndexs[1]
        tmp = body[index1]
        body[index1] = body[index2]
        body[index2] = tmp
    elif mutationOption == 4:
        select = random.randint(0, bodyLength-1)
        line = body[select]

        if ")" in line:
            tmp = line.index(")") + 1 # space right after u(p1...)
        else:
            tmp = line.index(" ")

        gatename = line[0:tmp].replace(" ","")
        datasec = line[tmp+1:].replace(" ","")
        newline = replaceGate(gatename, datasec)
        body[select] = newline
    elif mutationOption == 5:
        select = random.randint(0, bodyLength-1)
        line = body[select]

        if ")" in line:
            tmp = line.index(")") + 1 # space right after u(p1...)
        else:
            tmp = line.index(" ")

        gatename = line[0:tmp].replace(" ","")
        datasec = line[tmp+1:].replace(" ","")
        newline = replaceBits(gatename, datasec, bitDomain)
        body[select] = newline
    elif mutationOption == 6:
        select = random.randint(0, bodyLength-1)
        line = body[select]

        if ")" in line:
            tmp = line.index(")") + 1 # space right after u(p1...)
        else:
            tmp = line.index(" ")

        gatename = line[0:tmp].replace(" ","")
        datasec = line[tmp+1:].replace(" ","")
        newline = replaceInstruction(gatename, datasec, bitDomain)
        body[select] = newline


    with open(newFilePath, 'w') as newFile:
        newFile.write("\n//1 remove 2 insert 3 swap 4 replace gate 5 replace data 6 replace to a different instr \n//mutate option: " + str(mutationOption) + "\n")

        for x in header:
            newFile.write(x)
            newFile.write(";\n")
        for y in body:
            newFile.write(y)
            newFile.write(";\n")
        for z in footer:
            newFile.write(z)
            newFile.write(";\n")



def check_entanglement(newFilePath):
    _data = open(newFilePath).readlines()
    lines = []
    for physicalLine in _data:
        if physicalLine.lstrip().startswith("//"):
            continue
        logicLines = physicalLine.split(";")
        for logicLine in logicLines:
            logicLine = logicLine.strip()
            if not logicLine or logicLine == '\n':
                continue
            lines.append(logicLine)


    header = []
    footer = []
    body = []
    has_entanglement_gate = False
    for line in lines: # every line should be ended with a separator
        #print line

        if line.startswith("include ") or line.startswith("qreg ") or line.startswith("creg "):
            header.append(line)
        elif line.startswith("measure "): #TODO shall we also mutate this? currently no
            footer.append(line)
        else:
            body.append(line)
            if ")" in line:
                tmp = line.index(")") + 1 # space right after u(p1...)
            else:
                tmp = line.index(" ")

            gatename = line[0:tmp].replace(" ","")
            if gatename in binaryOPs or gatename in ternaryOPs:
                has_entanglement_gate = True
                return  has_entanglement_gate  # safe to return


    return  has_entanglement_gate



