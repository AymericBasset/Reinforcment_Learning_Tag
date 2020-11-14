import numpy as np
from Brain import Brain
import Parameters as p 

class Agent():
    def __init__(self,brain_spec=p.BRAIN_SPEC):

        #State and Physics init
        self.agent_reset()

        #Think_center init
        self.brain_spec = brain_spec
        self.brain = Brain(brain_spec)
  
    def agent_reset(self):

        #Physics
        self.pos = np.array([p.INIT_X,p.INIT_Y])
        self.velocity = np.array([0.0,0.0])
        self.acceleration = np.array([0,0])

        #State
        self.score = 0
        self.alive = True
        self.memory = []
        self.team = 1

        #Characteristics
        self.fov = p.FOV

    def update(self,world_input,reward):
        #add more maybe later
        action = np.argmax(self.think(world_input))
        self.moove(p.ACTION_DIC[action])
        self.add_reward(reward)
        self.remember(world_input)

    def moove(self,action):
        self.acceleration = action
        self.velocity += self.acceleration
        self.pos += self.velocity
        self.velocity *=  p.FRICTION_COEF
    
    def add_reward(self, reward):
        self.score += reward

    def think(self,world_input):
        return(self.brain.think(world_input))
    
    def mutate(self):
        self.brain.mutate()
    
    def die(self):
        self.alive = False
    
    def tag(self):
        self.team *= -1
    
    def remember(self,world_input):
        if self.memory:
            self.memory[-1][1] = self.score
        self.memory.append([world_input,self.score])

if __name__ == "__main__":
    agent = Agent(p.BRAIN_SPEC)
    print("Hello, I'm alive !")
    print(agent.__dict__)
    sensor = [1,1,0,0]
    reward = 1
    action_proba = agent.think(sensor)
    print("I'm thinking")
    print(action_proba)
    print("I should go there")
    print(p.ACTION_DIC[np.argmax(action_proba)])
    agent.update(sensor,reward)
    print("I mooved !")
    print(agent.__dict__)
    print("I'm thinking")
    print(action_proba)
    print("I should go there")
    print(p.ACTION_DIC[np.argmax(action_proba)])
    agent.update(sensor,reward)
    print("I mooved !")
    print(agent.__dict__)