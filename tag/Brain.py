import numpy as np
import random as rd


import tensorflow as tf
from tensorflow import keras


class Brain():
    def __init__(self,brain_spec, random = True, weights = None):

        self.brain_spec = brain_spec
        ##INIT 
        #This is a new brai,
        self.neurones = keras.Sequential()
        for i in range(len(brain_spec)-2):
            #init the weights between two layers, with matrix [layer_i,layer_i+1] and the bias
            self.neurones.add(keras.layers.Dense(brain_spec[i+1],activation= "elu",input_shape=(brain_spec[i],)))
        #output layer
        self.neurones.add(keras.layers.Dense(brain_spec[-1], activation="softmax"))

        #In case want specific value
        if not(random):
            assert(weights != None)
            self.neurones.set_weights(weights)
        
        #self.brain.compile(optimizer="adam", loss =t.tanh_custom_loss,metrics=[t.tanh_custom_loss])
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)



    def think(self, x):
        return(self.neurones(np.expand_dims(x,axis=0))).numpy()[0]
    

    def mutate(self,mutation_factor = 0.1):
        weights = self.neurones.get_weights()
        for layer in weights:
            layer += layer*rd.uniform(-1*mutation_factor,1*mutation_factor)*np.random.randint(2,size=layer.shape)
        self.neurones.set_weights(weights)
    
    def expand(self):
        pass
    
    def learn(self,memory):
        pass
        
        




if __name__ == "__main__":
    TEST = True
    if TEST:
        test_input = np.array([1,1,1,1])
        output_size = 4
        brain_spec = [test_input.shape[0],5,output_size]

        print("#################### RANDOM INIT ######################################")
        head = Brain(brain_spec,random = True)
        print(head.neurones.get_weights())
        print("#################### DEFINE INIT ######################################")
        head = Brain(brain_spec,random = False, weights=head.neurones.get_weights())
        print(head.neurones.get_weights())
        print(head.neurones.summary())
        print("#################### MUTATING ###########################################")
        head.mutate()
        print(head.neurones.get_weights()) 
        ##THINK 
        print("#################### THINKING ############################################")
        print(head.think(test_input))
        ##LEARN
        print(head.neurones.trainable_variables)
        print("#################### LEARNING ############################################")
        memory = [np.array([[1.0,1.0,10.0,10.0]]),np.array([2.0])]
        head.learn(memory)
