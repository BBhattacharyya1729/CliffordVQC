from circuit import *
class VariationalLayer:
    
    def __init__(self,nqubits,depth,clifford, initial_state = None):
        self.nqubits = nqubits
        self.depth = depth
        self.circuit = Circuit(nqubits,initial_state)
        self.clifford = clifford
        
    def __call__(self, parameters):
        p=np.array([np.split(i,3) for i in np.split(parameters,self.depth+1)])
        state = self.circuit.initial_state
        for i in range(self.depth+1):
            for j in range(self.nqubits):
                state = self.make_rx()(state, p[i][0][j],j)
                state = self.make_ry()(state, p[i][1][j],j)
                state = self.make_rz()(state, p[i][2][j],j)
            if(i<self.depth):
                for j in range(self.nqubits-1):
                    state = self.make_cx()(state,j,j+1)
        return state        

    def make_cx(self):
        def operation(state, qubit, target):
            return self.circuit.cx(state, qubit, target)
        return operation

    def make_rx(self):
        def operation(state,angle, qubit):
            if(self.clifford):
                return self.circuit.rx(state, np.pi/2 * np.round(angle* 2/np.pi) , qubit)
            return self.circuit.rx(state,  angle, qubit)
        return operation

    def make_ry(self):
        def operation(state, angle, qubit):
            if(self.clifford):
                return self.circuit.ry(state, np.pi/2 * np.round(angle * 2/np.pi) , qubit)
            return self.circuit.ry(state,  angle, qubit)
        return operation

    def make_rz(self):
        def operation(state, angle, qubit):
            if(self.clifford):
                return self.circuit.rz(state, np.pi/2 * np.round(angle * 2/np.pi) , qubit)
            return self.circuit.rz(state, angle, qubit)
        return operation
