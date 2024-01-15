import numpy as np
from featuremaps import * 
from variationalLayers import *
from qiskit.quantum_info import SparsePauliOp,Operator
from qiskit.circuit.library import EfficientSU2

import hypermapper
import json
import sys

def BayesianOptimizer(fun,num_params,samples,iterations,save_dir,name):
    """
    General purpose hypermapper based optimizer
    (Function) fun: Function to optimize over
    (Int) num_params: The number of parameters
    (Int) iterations: The number of iterations
    (String) save_dir: text name of the save directory
    (String) name: name for all the logging
    """
    
    hypermapper_config_path = save_dir + "/"+name+"_hypermapper_config.json"
    config = {}
    config["application_name"] = "cafqa_optimization_"+name
    config["optimization_objectives"] = ["value"]
    config["design_of_experiment"] = {}
    config["design_of_experiment"]["number_of_samples"] = samples
    config["optimization_iterations"] = iterations
    config["models"] = {}
    config["models"]["model"] = "random_forest"
    config["input_parameters"] = {}
    config["print_best"] = True
    config["print_posterior_best"] = True
    for i in range(num_params):
        x = {}
        x["parameter_type"] = "ordinal"
        x["values"] = [0, 1, 2, 3]
        x["parameter_default"] = 0
        config["input_parameters"]["x" + str(i)] = x
    config["log_file"] = save_dir + '/'+name+'_hypermapper_log.log'
    config["output_data_file"] = save_dir + "/"+name+"_hypermapper_output.csv"
    with open(hypermapper_config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)
    stdout=sys.stdout
    with open(save_dir+"/"+name+'_optimizer_log.txt', 'w') as sys.stdout:
        hypermapper.optimizer.optimize(hypermapper_config_path,fun)
    sys.stdout = stdout
    
    fun_ev = np.inf
    x = None
    with open(config["log_file"]) as f:
        lines = f.readlines()
        counter = 0
        for idx, line in enumerate(lines[::-1]):
            if line[:16] == "Best point found" or line[:29] == "Minimum of the posterior mean":
                counter += 1
                parts = lines[-1-idx+2].split(",")
                value = float(parts[-1])
                if value < fun_ev:
                    fun_ev = value
                    x = [int(y) for y in parts[:-1]]
            if counter == 2:
                break
    return fun_ev, x


class Classifier:
    def __init__(self, nqubits, x_len, bitstring, layer_depth):
        
        self.nqubits = nqubits
        self.x_len = x_len
        self.bitstring= bitstring
        self.depth = layer_depth
        self.iterations = 0 
        
        
    def get_state(self, data, parameters,clifford=False):
        state = FeatureMap(self.nqubits, self.x_len,clifford)(self.bitstring, data.T)
        state = VariationalLayer(self.nqubits,self.depth,clifford, initial_state = state)(parameters)
        return state
    
    def get_prediction(self,data,parameters,noise=None,clifford=False):
        if(noise == None):
            predictions = []    
            state = self.get_state(data,parameters,clifford)
            for i in state:
                s=0
                for j,v in enumerate(i):
                    s+=(abs(v) ** 2 )* (-1)**list(bin(j)[2:]).count('1') 
                predictions.append(s.real)
        else:
            encoder = string_to_circuit(self.bitstring, self.nqubits, self.x_len)[0]
            circuits = [encoder.assign_parameters(x[:encoder.num_parameters]) for x in X]
            layer = EfficientSU2(num_qubits=self.nqubits,reps=self.depth,su2_gates=['rx','ry','rz'],entanglement='linear').assign_parameters(parameters)
            circuits = [c.compose(layer) for c in circuits]
            [c.save_density_matrix() for c in circuits]
            Z = SparsePauliOp('Z' * self.nqubits)
            noisy_simulator = AerSimulator(method='density_matrix', noise_model = noise_model)
            return [noisy_simulator.run(c).result().data()['density_matrix'].expectation_value(Z).real for c in circuits]
            
        #     state = noisy_states(s=self.bitstring, M=self.nqubits, x_len=self.x_len,X=data,noise_model=noise)
        #     op = Operator(EfficientSU2(num_qubits=self.nqubits,reps=self.depth,su2_gates=['rx','ry','rz'],entanglement='linear').assign_parameters(parameters))
        #     state = [s.evolve(op) for s in state]
        #     Z = SparsePauliOp('Z' * self.nqubits)
        #     for i in state:
        #         predictions.append(i.expectation_value(Z).real)
        # return predictions
    
    def get_loss(self,X,y,parameters,noise=None,callback=None,clifford=False):
        preds = self.get_prediction(X,parameters,noise,clifford)
        
        s=0
        for i,v in enumerate(y):
            s+=(preds[i]-v)**2/len(y)
        #self.iterations+=1
        if(callback != None):
            callback(parameters,s)
        return s
    
    def get_accuracy(self,X,y,parameters,noise=None,clifford=False):
        preds = self.get_prediction(X,parameters,noise,clifford)
        c=0
        for i,v in enumerate(y):
            if(preds[i]*v>=0):
                c+=1
        return c/len(y)
    
    def get_clifford_loss(self,X,y,inputs):
        params = np.array(list(inputs.values())) * np.pi/2
        return self.get_loss(X,y,params,clifford=True)
        
    def run_clifford_optimization(self,X,y,samples,iterations,save_dir,name):
        v, p = BayesianOptimizer(lambda inputs: self.get_clifford_loss(X,y,inputs),self.nqubits * (self.depth+1) * 3,samples,iterations,save_dir,name)
        return v, np.array(p) * np.pi/2
    
    
    
    
    def run_optimization(self,optimizer, X, y,log_file,initial_point,noise=None):
        
        history = {'parameter_list':[],'cost_list':[]}
        def callback(parameters,loss):
            history['parameter_list'].append(parameters)
            history['cost_list'].append(loss)
        
        def cost(parameters):
            self.iterations+=1
            loss = self.get_loss(X,y,parameters,noise=noise,callback=callback,clifford=False)
            if(self.iterations % 100==0):
                    print("Iterations " + str(self.iterations),file=open(log_file, 'a'))
                    print("Loss " + str(loss),file=open(log_file, 'a'))
                    if(noise != None):
                        true_loss = self.get_loss(X,y,parameters,noise=None,callback=callback,clifford=False)
                        print("True Loss " + str(loss),file=open(log_file, 'a'))
                  
                
            return loss
        
        self.iterations = 0
        result = optimizer.minimize(cost, x0=initial_point, bounds=[[0,2*np.pi]]*len(initial_point))
        
        return (result.x,result.fun,history)
 
