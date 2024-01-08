import numpy as np
from numpy import pi as π
from circuit import *

def layers(s):
    l = ["".join(str(i) for i in s[7*i:7*i+7]) for i in range(len(s)//7)]
    return l

class FeatureMap:

    def __init__(self, nqubits, nparameters,clifford,initial_state = None):
        self.gates = gates = {}
        for suffix in ['0000','0001','0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111']:
            angle = np.pi/8 * int(suffix,2)
            gates['000'+suffix] = self.make_h()
            gates['001'+suffix] = self.make_cx()
            
            if(suffix != '0000'):
                gates['101'+suffix] = self.make_rx(angle)
                gates['111'+suffix] = self.make_rz(angle)
                gates['110'+suffix] = self.make_ry(angle)
            else:
                gates['101'+suffix] = self.make_id()
                gates['111'+suffix] = self.make_id()
                gates['110'+suffix] = self.make_id()
                

            gates['010'+suffix] = self.make_id()
            gates['011'+suffix] = self.make_id()
            gates['100'+suffix] = self.make_id()
        self.nqubits = nqubits
        self.nparameters = nparameters
        self.circuit = Circuit(nqubits,initial_state)
        self.clifford = clifford

    def __call__(self, s, data):
        k = 0
        state = self.circuit.initial_state
        for ndx, z in enumerate(layers(s)):
            qubit = ndx % self.nqubits
            target = (ndx + 1) % self.nqubits
            fn = self.gates[z]
            state, k = fn(state, data, k, qubit, target)
        if k == 0 and data.ndim == 2:
            state = np.ones((data.shape[1],1)) * state.reshape(1,-1)
        return state

    def make_id(self):
        def operation(state, data, k, qubit, target):
            return state, k
        return operation

    def make_h(self):
        def operation(state, data, k, qubit, target):
            return self.circuit.h(state, qubit), k
        return operation

    def make_cx(self):
        def operation(state, data, k, qubit, target):
            return self.circuit.cx(state, qubit, target), k
        return operation

    def make_rx(self, angle):
        def operation(state, data, k, qubit, target):
            ndx = k % self.nparameters
            if(self.clifford):
                return self.circuit.rx(state, np.pi/2 * np.round(data[ndx,:]*angle * 2/np.pi) , qubit), k+1
            return self.circuit.rx(state,  data[ndx,:]*angle, qubit), k+1
        return operation

    def make_ry(self, angle):
        def operation(state, data, k, qubit, target):
            ndx = k % self.nparameters
            if(self.clifford):
                return self.circuit.ry(state, np.pi/2 * np.round(data[ndx,:]*angle * 2/np.pi) , qubit), k+1
            return self.circuit.ry(state,  data[ndx,:]*angle, qubit), k+1
        return operation

    def make_rz(self, angle):
        def operation(state, data, k, qubit, target):
            ndx = k % self.nparameters
            if(self.clifford):
                return self.circuit.rz(state, np.pi/2 * np.round(data[ndx,:]*angle * 2/np.pi) , qubit), k+1
            return self.circuit.rz(state,  data[ndx,:]*angle, qubit), k+1
        return operation

def cost(s,M):
    N = len(s)//(7*M)
    layers = np.array([s[7*i:7*i+7] for i in range(len(s)//7)],dtype=str)
    layers = np.split(layers,N)
    
    non_local=0
    gate = 0
    h=0
    for st in layers:
        for j in st:
            s="".join(i for i in j)
            if(s[:3]== '000'):
                h+=1
                gate+=1
            elif(s[:3]== '001'):
                gate+=1
                non_local+=1
            elif(s[:3] == '101' and s[-4:] != '0000'):
                gate+=1
            elif(s[:3] == '110' and s[-4:] != '0000'):
                gate+=1
            elif(s[:3] == '111' and s[-4:] != '0000'):
                gate+=1
    return gate+4*non_local+h


from qiskit.circuit import QuantumCircuit,ParameterVector

def string_to_circuit(s, M, x_len):
    """
    Convert an array of 0's and 1's into a circuit
    
    (Iterable[String]) s: Sequence of bits
    (Int) M: number of qubits
    (Int) x_len: Dimension of data
    """
    X  = ParameterVector('x',x_len)
    qc = QuantumCircuit(M)
    N = len(s)//(7*M)
    layers = np.array([s[7*i:7*i+7] for i in range(len(s)//7)],dtype=str)
    layers = np.split(layers,N)
    
    index = 0
    gates = 0
    non_locals = 0
    h=0
    for n,instr in enumerate(layers):
        for i in range(M):
            s=instr[i]
            param,local,gate,bool_h=modify(qc,s,i,X[index])
            if(gate):
                gates+=1
                if(param):
                    index = (index+1) % x_len
                elif(not local):
                    non_locals+=1
                elif(bool_h):
                    h+=1
    return qc,gates,non_locals,h

def modify(qc,st,qubit,parameter):
    '''
    Add a specified operator to existing circuit
    (QuantumCircuit) qc: Quantum Circuit to modify
    (String) st: String for operator
    (Int) qubit: qubit index
    (Parameter): the parameter
    
    Returns None
    '''
    N=qc.num_qubits
    param = False
    local=True
    gate = True
    h=False
    s="".join(i for i in st)
    
    if(s[:3]== '000'):
        qc.h(qubit)
        h=True
    elif(s[:3]== '001'):
        qc.cnot(qubit, (qubit+1) % N)
        local=False
    elif(s[:3] == '101' and s[-4:] != '0000'):
        qc.rx(np.pi/8 * int(s[-4:],2) * parameter ,qubit)
        param= True
    elif(s[:3] == '110' and s[-4:] != '0000'):
        qc.ry(np.pi/8 * int(s[-4:],2) * parameter ,qubit)
        param= True
    elif(s[:3] == '111' and s[-4:] != '0000'):
        qc.rz(np.pi/8 * int(s[-4:],2) * parameter,qubit)
        param= True
    else:
        gate = False
    return param,local,gate,h


from qiskit_aer import AerSimulator

def noisy_states(s, M, x_len,X,noise_model):
    encoder = string_to_circuit(s, M, x_len)[0]
    circuits = [encoder.assign_parameters(x[:encoder.num_parameters]) for x in X]
    [c.save_density_matrix() for c in circuits]
    noisy_simulator = AerSimulator(method='density_matrix', noise_model = noise_model)
    return [noisy_simulator.run(c).result().data()['density_matrix'] for c in circuits]