from random import randint
import numpy as np

rng = np.random  
class ReinforceSmall():
    def xavier_init(self, fan_in, fan_out):
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_out, fan_in))
    def __init__(self, D:int,H:int,H1:int,H2:int,O:int):
        self.model = {
            "W1": self.xavier_init(D, H),  # Input => hidden weights
            "W2": self.xavier_init(H, H1),
            "W3": self.xavier_init(H1, H2),
            "W4": self.xavier_init(H2, O)
        }
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def forwardPropagation(self, streets:list[4], currState, iter, pos, dimensions):
        inputs = []
        for i in streets:
            inputs.append(i)
        for i in currState:
            inputs.append(i)
        inputs.append(iter/1000)
        inputs.append(pos[0]/dimensions[0])
        inputs.append(pos[1]/dimensions[1])
        inputs = np.array(inputs)

        h1 = np.dot(self.model["W1"],inputs)
        h1[h1 < 0] *= 0.1
        h2 = np.dot(self.model["W2"], h1)
        h2[h2 < 0] *= 0.1
        h3 = np.dot(self.model["W3"], h2)
        h3[h3 < 0] *= 0.1
        o = np.dot(self.model["W4"], h3)
        o = list(map(self.sigmoid, o))
        return o



smallReinforce = ReinforceSmall(10, 20, 10, 5, 3)

def decide(dimensions, city, pos, currState, time:int, method:str="randomD"):
    global smallReinforce
    match method:
        case "random":
            return random()
        case "oppositeWithTurns":
            return alwaysChangeWithTurns(currState)
        case "smallReinforce":  
            s1 = city[pos[0]*2, pos[1]] if pos[1] < dimensions[1] else 0
            s2 = city[pos[0]*2, pos[1]-1] if pos[1]-1 >= 0 else 0
            s3 = city[pos[0]*2-1, pos[1]] if pos[0]*2-1 >= 0 else 0
            s4 = city[pos[0]*2+1, pos[1]] if pos[0]*2+1 < dimensions[0] else 0
            aprob = smallReinforce.forwardPropagation([s1, s2, s3, s4], currState, time, pos, dimensions)
            actions = list(map(lambda a: 1 if rng.uniform() < a else 0, aprob))
            return actions

def random():
    return [randint(0,1),randint(0,1),randint(0,1)]

def alwaysChangeWithTurns(currState):
    return (not currState[0], 1, 1)
