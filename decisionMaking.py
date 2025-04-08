from random import randint

class ReinforceSmall():
    def decide(streets:list[4], iter, pos):
        pass

smallReinforce = ReinforceSmall()

def decide(city, pos, currState, time:int, method:str="randomD"):
    match method:
        case "random":
            return random()
        case "oppositeWithTurns":
            return alwaysChangeWithTurns(currState)
        case "smallReinforce":
            smallReinforce.decide()

def random():
    return [randint(0,1),randint(0,1),randint(0,1)]

def alwaysChangeWithTurns(currState):
    return (not currState[0], 1, 1)
