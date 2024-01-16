from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pygad
from featuremaps import *
from QSVM import *
import numpy as np

def accuracy_metric(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)
    return(accuracy)

def Dataset(X, y, test_size_split=0.3):
    train_sample, test_sample, train_label, test_label = train_test_split(X, y, stratify=y, test_size=test_size_split)
    return train_sample, train_label, test_sample, test_label

def check(temp_train_y,n):
    vals = np.split(temp_train_y,len(temp_train_y)//n)
    return np.all([(list(i).count(-1)>=2 and list(i).count(1)>=2) for i in vals])

class CliffordOptimizer:
    
    def fitness_func(self,ga_instance, solution, solution_idx):
        training_features, training_labels, test_features, test_labels = Dataset(self.train_x, self.train_y)
        model = QSVM(lambda parameters: FeatureMap(self.M,self.x_len,True)(solution, parameters),training_features, training_labels)
        y_pred = model.predict(test_features)
        acc =  accuracy_metric(test_labels, y_pred)
        weight = cost(solution,self.M)
        
        return [-weight* (1+acc**2),acc]
    
    def on_generation(self,ga_instance):
        print(f"Generation = {self.ga_instance.generations_completed}",file=open(self.log_file, 'a', encoding='utf-8'))
        print(f"Fitness    = {self.ga_instance.best_solution(pop_fitness=self.ga_instance.last_generation_fitness)[1]}",file=open(self.log_file, 'a',encoding='utf-8'))
        print(f"Change     = {self.ga_instance.best_solution(pop_fitness=self.ga_instance.last_generation_fitness)[1] - self.last_fitness}",file=open(self.log_file, 'a', encoding='utf-8'))
        if(self.ga_instance.generations_completed % 5 == 0):
            qc=string_to_circuit(self.ga_instance.best_solution()[0],self.M,self.x_len)[0]
            print((qc.draw()),file=open(self.log_file, 'a', encoding='utf-8'))
            print("--------------------",file=open(self.log_file, 'a', encoding='utf-8'))
        self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    

    
    def __init__(self,M, N, x_len, train_x,train_y, log_file, n=None, num_generations = 1000, num_parents_mating = 10, sol_per_pop=30,crossover_probability=0.7,mutation_probability=0.2):
        
        self.x_len = x_len
        self.M = M
        self.N = N
        
        
        self.train_x=train_x
        self.train_y=train_y
        self.last_fitness = 0
         
        if(n==None):
            self.n=len(self.train_x)
        else:
            self.n=n  
          
        self.log_file = log_file
        
        self.ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=self.M * self.N * 7,
                       fitness_func=self.fitness_func,
                       on_generation=self.on_generation,
                       crossover_probability=crossover_probability,
                       mutation_probability=mutation_probability,
                      gene_type=int,
                      gene_space = [0,1],
                      parent_selection_type='nsga2',
                      )
        
    

    def run(self):
        self.ga_instance.run()
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution(self.ga_instance.last_generation_fitness)
        return solution, solution_fitness

def noisy_kernel_acc(s, M, x_len,noise_model,train_x,test_x,train_y,test_y):
    encoder = string_to_circuit(s,M,x_len)[0]
    f = lambda x: [encoder.assign_parameters(X[:encoder.num_parameters]) for X in x]
   
    q = noisy_QSVM(f,train_x,train_y,noise=noise_model)
    
    train_pred  = q.predict(train_x)
    train_acc =  accuracy_metric(train_pred, train_y)
    
    test_pred  = q.predict(test_x)
    test_acc =  accuracy_metric(test_pred, test_y)
    return (train_acc,test_acc)

def kernel_acc(s, M, x_len,train_x,test_x,train_y,test_y):
    f = lambda x: FeatureMap(M,x_len,False)(s, x)
    q = QSVM(f,train_x,train_y)
    
    train_pred  = q.predict(train_x)
    train_acc =  accuracy_metric(train_pred, train_y)
    
    test_pred  = q.predict(test_x)
    test_acc =  accuracy_metric(test_pred, test_y)
    return (train_acc,test_acc)
