from random import randint

def decide(city, pos, currState, time:int, method:str="randomD"):
    match method:
        case "random":
            return random()
        case "oppositeWithTurns":
            return alwaysChangeWithTurns(currState)

def random():
    return [randint(0,1),randint(0,1),randint(0,1)]

def alwaysChangeWithTurns(currState):
    return (not currState[0], 1, 1)