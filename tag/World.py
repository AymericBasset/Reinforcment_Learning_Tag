import Agent
import Parameters as p

class World():
    """
    This class handle all the entities placement and the physics of the game
    """
    def __init__(self, agents, random = True):

        self.n_agents = p.N_AGENTS
        
        self.map_entities = []
        self.map_signals = []
        self.map_resources = []
        self.map = [self.map_entities,self.map_signals,self.map_resources]


        if random:
            self.agents = agents
        else:
            self.agents = [Agent() for i in range(self.n_agents)]
    
    def get_input(self,agent):
        return(self.map[:,agent.pos[0]-agent.fov:agent.pos[0]+agent.fov+1,agent.pos[1]-agent.fov:agent.pos[1]+agent.fov+1])