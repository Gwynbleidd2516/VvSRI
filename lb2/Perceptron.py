from Neuron import *
import numpy as np

class Perceptron:
    def __init__(self, Lambda, tolerence):
        self.Lambda=Lambda
        self.tolerence=tolerence
        
        self.layers=[[Neuron(), Neuron()], [Neuron()]]
        # self.inputNeurons=[Neuron, Neuron]
        # self.outputNeuron=Neuron
        
        self.weights=np.array([np.array([np.random.random() for _ in range(2)]) for _ in range(2)])
        
        self.train_mass=np.array([0])
        self.train_ans=np.array([0])
        self.ans=np.array([0])
        pass
    
    def calc(self, inp:np.array):
        for i in range(len(self.layers[0])):
            self.layers[0][i].val=self.weights[0][i]*inp[i]
        
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                ii=np.sum(self.layers[i-1][k].val*self.weights[i][k] for k in range(len(self.layers[i-1])))
                self.layers[i][j].setValue(ii)
        return self.layers[::-1][0][0].val
    
    def learn(self, train:np.array, train_ans:np.array, iterations:int):
        for i in range(iterations):
            ans=[]
            for x in train:
                ans.append(self.calc(x))
            grad=sum([(ans[j]-train_ans[j])**2 for j in range(len(train_ans))])
            print(f'Iteration {i}: ', f'Error {grad}')
            if(grad<=self.tolerence):
                break
            self.weights-=self.Lambda*grad
            
            
        
        
    
    