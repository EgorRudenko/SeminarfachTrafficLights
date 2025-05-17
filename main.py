import pygame
import numpy as np
from random import randint, seed
from decisionMaking import decide
from decisionMaking import learn


decisionMethod = "oppositeWithTurns"       # how the traffic light's state is decided
graphicMode = True
showTrafficLights = False
'''
Options for now:
"random"
"oppositeWithTurns": horisontal to vertical, vertical to horizontal with turns always allowed
"smallReinforce"
'''

trafficLightWidth = 5
mem = 10000
generateMoreCars = True # whether new cars should appear while simulation
carsToGenerate = 20
trafficLightBaseState =[1,1,1]      # doesn't actually change anything (how trafficLight is at the beginning plays almost no role later)
carsForRed = 5                      # how many cars on a street for it to appear red



class console_colors():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m' # orange on some systems
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    LIGHT_GRAY = '\033[37m'
    DARK_GRAY = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    WHITE = '\033[97m'

    RESET = '\033[0m' # called to return to standard terminal text color

con_cols = console_colors()
rng = np.random.default_rng(seed = 1)
seed(1)

class Street():
    cars = 0
    carIndeces = []
    toGo = True

class Car():
    arrivalTime = None
    def __init__(self, x:int, y:int, xDest:int, yDest:int, i:int, birthTime:int):
        global streetStates
        self.x = x
        self.y = y
        self.path = [(x, y)]
        self.birthTime = birthTime
        self.timeToGo = 0           # after arriving at street you have to wait before moving further, if negative => we waited more than we could
        self.xDest = xDest
        self.yDest = yDest
        self.queue = streetStates[y][x].cars # useless right now, as far as i can tell
        streetStates[y][x].cars += 1
        streetStates[y][x].carIndeces.append(i)
    def updateCoord(self, a:list, i:int, trafficLight)->None:
        global streetStates, check, rewards
        if streetStates[self.y][self.x].carIndeces[0] == i and streetStates[self.y][self.x].toGo and self.timeToGo <= 0:
            # reward depends on amount of unnecesary time I had to wait (because there were cars in front of me)
            #rewards[iter][trafficLight[0]][trafficLight[1]] += 5/-(cars[i].timeToGo-1)
            #rewards[iter][trafficLight[0]][trafficLight[1]] += 20
            streetStates[self.y][self.x].cars -= 1
            streetStates[self.y][self.x].carIndeces.pop(0)
            self.x, self.y = a[0], a[1]
            self.path.append((a[0], a[1]))
            streetStates[self.y][self.x].carIndeces.append(i)
            self.queue = streetStates[self.y][self.x].cars
            streetStates[self.y][self.x].cars += 1
            streetStates[self.y][self.x].toGo = False
            self.timeToGo = 5
            retMsg = f"success, carIndeces[0] = {streetStates[self.y][self.x].carIndeces[0]}, i = {i}, toGo = {streetStates[self.y][self.x].toGo}, self.timeToGo = {self.timeToGo}"
        else:
            retMsg =  f"faliure, carIndeces[0] = {streetStates[self.y][self.x].carIndeces[0]}, i = {i}, toGo = {streetStates[self.y][self.x].toGo}, self.timeToGo = {self.timeToGo}"
        self.timeToGo -= 1
        #self.timeToGo = 0 if self.timeToGo < 0 else self.timeToGo
        return retMsg

streetStates = []
cars = []
carsWhichArrived = []
iter = 1 # artificial time (number of go() iterations)
decisionsMade = []      # descision matrix is a turple of matricies: (numbers of cars, trafficLights, descision/new traffic light state)
activeCars = 0
running = True
citySize = (11,6) # width, heigth in trafficLight. 11 is height however visually it looks like 5 because both vertical and horizontal are there
city = np.zeros((citySize[0],citySize[1]+1)) # city[row][column] used primarily for coloring
cityInput = np.zeros((citySize[0],citySize[1]+1)) # used for AI (how many cars are there on each street)
trafficLights = np.zeros((citySize[0]//2+1,citySize[1]+1, 3)) #  trafficLights[row][column][parameter]
rewards = np.zeros((mem,citySize[0]//2+1,citySize[1]+1)) 

#print(rewards)
for i in range(citySize[0]):
    if i % 2 == 0:
        city[i][citySize[1]] = None

for i in range(citySize[1]+1):
    for j in range(citySize[0]//2+1):
       trafficLights[j][i] = trafficLightBaseState


for i in range(citySize[0]):
    streetStates.append([])
    for j in range(citySize[1]+1):
        if (j <= citySize[1]) or (i % 2 == 1):
            streetStates[i].append(Street())

def initialize():
    global decisionsMade, cars, carsWhichArrived, iter, running, carsToGenerate, activeCars, rewards
    activeCars = 0
    decisionsMade = []
    iter = 1
    cars = []
    carsWhichArrived = []
    running = True
    rewards = np.zeros((mem,citySize[0]//2+1,citySize[1]+1))
    genCars(carsToGenerate)


def normalizeCoordinates(a:int, maxA:int) -> int: # normalize (theorethical [-infinity; +infinity] from normal distribution -> [0; maxA]) + round the result
    if a > maxA:
        return maxA
    if a < 0:
        return 0
    return int(a)

def genCars(n:int, iter:int = 0) -> None:
    global activeCars, carsWhichArrived
    for i in range(n):
        x, y = rng.multivariate_normal([2,2], [[1,0],[0,1]], 1)[0]
        y = normalizeCoordinates(y, citySize[0]-1)
        x = normalizeCoordinates(x,  citySize[1]-1) if y%2 == 0 else normalizeCoordinates(x,  citySize[1])
        xDest, yDest = rng.multivariate_normal([7,7], [[10,0],[0,1]], 1)[0]
        yDest = normalizeCoordinates(yDest, citySize[0]-1)
        xDest = normalizeCoordinates(xDest,  citySize[1]-1) if yDest%2 == 0 else normalizeCoordinates(xDest,  citySize[1])

        probToSwitch = randint(0,1000)
        if (1000-iter) < probToSwitch:
            temp = yDest
            yDest = y
            y = temp
            
            temp = xDest
            xDest = x
            x = temp
        cars.append(Car(x, y, xDest, yDest, len(cars), iter))
        activeCars += 1

def pss():  # print streets states
    for i in range(citySize[0]):
        for j in range(citySize[1]+1):
            if (j <= citySize[1]) or (i % 2 == 1):
                print(streetStates[i][j].cars, end=" ")
        print("\n")

def lightToDir(l:list[int]) -> list[list[int]]:  # transform light state e.g [0,0,1] to directions possible [right, left, down, up]
    v, h = [0,0,0,0], [0,0,0,0] # right, left, down, up
    if l[0] == 1: # cars on horisontal streets
        h[0], h[1] = 1,1    # right + left
        if l[1] == 1: # if second parameter then down possible
            h[2] = 1
        if l[2] == 1: # if third parameter then up is possible
            h[3] = 1
    else:   # cars on vertical streets
        v[2], v[3] = 1,1    # up + down
        if l[1] == 1:   # if second parameter then right is possible
            v[0] = 1
        if l[2] == 1:   # if third parameter then left is possible
            v[1] = 1
    return [h,v]

# legendary piece of shitcode
def move(x:int, y:int, DestX:int, DestY:int) -> list[int]:
    if x == DestX and y == DestY: return ["already there", None]
    if y%2 == 1:    # we are on a vertical line
        directionY = np.sign(DestY-y)
        directionX = np.sign(DestX-x)
        diffY = DestY-y
        diffX = DestX-x
        trafficLight = directionY+(directionY==0)*(randint(0,1)*2-1)    # which traffic light to use more in case of dual possible choice or just direction
        canMove = lightToDir(trafficLights[y//2+trafficLight][x])[1]      # possible movements according to traffic light
        if canMove[(directionX<0)*1] and canMove[(directionY>0)*2+(directionY<0)*3] and abs(diffY) > 1 and abs(diffX) > 1:  # if both directions are possible and make sense, we choose randomly
            r = randint(0,1)
            return [[x+directionX*(directionX<0)*r,y+trafficLight*r+(not r)*2*directionY], [y//2+trafficLight, x]]
        elif canMove[(directionY>0)*2+(directionY<0)*3] and (abs(diffY)>2 or diffX == 0) and abs(diffY) != 1:    # movement along vertical is rational and allowed
            return [[x,y+2*directionY], [y//2+trafficLight, x]]
        elif canMove[(directionX<0)*1] and (abs(diffY) <= 2 or diffX > 1):      # movement to the horizontal is rational and allowed 
            return [[x+directionX*(directionX<0), y+trafficLight], [y//2+trafficLight, x]]
        else:   # traffic light forbids both
            return [None, [y//2+trafficLight, x]]

    else:           # car is currently on horizontal street
        directionY = np.sign(DestY-y)
        directionX = np.sign(DestX-x)
        diffX = DestX-x
        diffY = DestY-y
        trafficLight = directionX+(directionX==0)*((randint(0,1)*2-1)*(DestY%2==0)) + (directionX==0)*(-1)*(DestY%2==1)  # random one if the street is directly under car
        canMove = lightToDir(trafficLights[y//2][x+trafficLight*(trafficLight>0)])[0]
        if directionY == 0:     # only movement in x direction is needed
            if canMove[0+1*(directionX<0)]:
                return [[x+directionX, y], [y//2, x+trafficLight*(trafficLight>0)]]
            else:
                return [None, [y//2, x+trafficLight*(trafficLight>0)]]
        else:
            if canMove[0+1*(directionX<0)] and canMove[2*(directionY>0)+3*(directionY<0)] and abs(diffX) > 1 and abs(diffY) > 1:
                r = randint(0,1)
                return [[x+directionX*r+(not r)*(trafficLight>0), y+(not r)*directionY], [y//2, x+trafficLight*(trafficLight>0)]]
            elif canMove[0+1*(directionX<0)] and ((DestY%2 == 1 and (diffX > 1 or diffX < 0)) or (DestY%2 == 0 and (not directionY or abs(diffX) > 1))):
                return [[x+directionX, y], [y//2, x+trafficLight*(trafficLight>0)]]
            elif canMove[2*(directionY>0)+3*(directionY<0)] and ((DestY%2 == 1 and ( (diffX == 1 or diffX == 0) or abs(diffY > 1) )) or DestY%2 == 0): 
                return [[x+1*(trafficLight>0), y+directionY], [y//2, x+trafficLight*(trafficLight>0)]]
            else:
                return [None, [y//2, x+trafficLight*(trafficLight>0)]]

def color(x:int):
    greenPower = 150 # 0- 255
    return (x*(x<0.5)*510+255*(x>=0.5),(255-(x-0.5)*510)*(x>=0.5)+greenPower*(x<0.5),0)


def go(): # move cars according to the rules
    global cars, streetStates, city, citySize, iter, activeCars, carsForRed
    max = carsForRed     # used for coloring (what is considered red)
    iter += 1

    doWeGenerate = randint(0, 250000)
    if (iter - 500)**2 > doWeGenerate and iter < 1000 and generateMoreCars:
        genCars(randint(0, 1), iter)

    for i in range(len(cars)):
        if cars[i] != "dummy":
            newCoord = move(cars[i].x, cars[i].y, cars[i].xDest, cars[i].yDest)
            # New Coordinate is either None or "already there" or [newCoordinate, traffic light responsible for the decsision]
            newCoord, trafficLight = newCoord[0], newCoord[1]
            if newCoord != None and newCoord != "already there":
                cars[i].updateCoord(newCoord, i, trafficLight)
            elif newCoord == None:
                if streetStates[cars[i].y][cars[i].x].carIndeces[0] == i and streetStates[cars[i].y][cars[i].x].toGo:
                    #rewards[iter][trafficLight[0]][trafficLight[1]] -= 2
                    pass
                cars[i].timeToGo -= 1
            elif newCoord == "already there":
                #print(i)
                streetStates[cars[i].y][cars[i].x].cars -= 1
                streetStates[cars[i].y][cars[i].x].carIndeces.remove(i)
                carsWhichArrived.append(cars[i])
                carsWhichArrived[-1].arrivalTime = iter
                cars[i] = "dummy"
                #print("index ", i)
                activeCars -= 1
    for i in range(citySize[0]):
        for j in range(citySize[1]+1):
            if (j <= citySize[1]) or (i % 2 == 1):
                if max != 0: 
                    streetStates[i][j].toGo = True
                    city[i][j] = (streetStates[i][j].cars/max) if (streetStates[i][j].cars/max) <= 1 else 1
                    cityInput[i][j] = streetStates[i][j].cars


running_average = 800
decay = 0.99
running_average_max = 0

def logic():
    global iter, activeCars, carsWhichArrived, activeCars, trafficLights, cityInput, running, con_cols, running_average, decay, running_average_max, rewards
    
    aboutDescision = []

    # 3 Inputs
    aboutDescision.append(iter)
    aboutDescision.append(cityInput)
    aboutDescision.append(trafficLights)
    for j in range(citySize[0]//2+1):
        for i in range(citySize[1]+1):
            des = decide(citySize, cityInput, (j,i), trafficLights[j][i], iter, method=decisionMethod)
            if des[0] != trafficLights[j][i][0]:
                rewards[iter][j][i] += 10
            if des[1] == 1:
                rewards[iter][j][i] += 1
            if des[2] == 1:
                rewards[iter][j][i] += 1
            trafficLights[j][i] = (des[0], des[1], des[2])

            
    # 1 output
    aboutDescision.append(trafficLights)

    decisionsMade.append(aboutDescision)


    if (iter < 1000 or activeCars > 0) and iter < mem-500:
        go()
    else:
        running = False
    
    if iter % 100 == 0:
        learn(rewards, mode = "vocal", model=decisionMethod)

    if iter % 1000==0:
        print("I'm still there")
    if iter % 100 == 0 or (activeCars == 0 and iter > 1000) or iter >= mem-500:
        s = 0
        #print(activeCars, iter, iter < 1000 or activeCars > 0, running)
        if activeCars == 0 and iter > 1000:
            activeCars = -1             # I just want to get one message
        max = 0
        min = 1000000
        for i in carsWhichArrived:
            travelTime = i.arrivalTime - i.birthTime
            s += travelTime
            if travelTime < min:
                min = travelTime
            if travelTime > max:
                max = travelTime
        try:
            average = s/len(carsWhichArrived)
            median = (min+max)/2
        except:
            average = "division by 0"
            median = "does not exist yet"
        if (activeCars == -1 and iter > 1000):
            running_average = running_average*decay + average*(1-decay)
            if running_average > running_average_max:
                running_average_max = running_average
            print(f"{con_cols.RED}Running average: {running_average}")
            print(f"Max running average: {running_average_max}")
            print(f"Current iteration average: {average}{con_cols.RESET}\nCurrent iteration median: {median}\nCurrent iteration max: {max}")
            #print(trafficLights)
            learn(rewards, model=decisionMethod)
        else:
            try:
                pass
                #print(f"Average: {average}\nMedian: {median}\nMax: {max}")
            except:
                pass
                #print("Not sure what happened, but it probably is so bad, that after 100 iterations no car arrived")


def pygameAnimation():
    global running
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()

    time = 0 # for drawing output (frame frequency)
    lengthOfAFrame = 0

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        time += clock.get_time()
        if time >= lengthOfAFrame:
            logic()
            time = 0
        
        # Draw stuff
        screen.fill((20,20,20))
        for i in range(citySize[0]+1):
            for j in range(citySize[1]+1):
                if i != citySize[0]:
                    if i % 2 == 1:
                        pygame.draw.line(screen, color(city[i][j]),(5+j*30,5+(i//2)*30), (5+j*30,35+(i//2)*30), 2)
                    elif j != citySize[1]:  # horizontal
                        pygame.draw.line(screen, color(city[i][j]),(5+j*30,5+(i//2)*30), (35+j*30,5+(i//2)*30), 2)
                # do the traffic lights
                if i % 2 == 1 and showTrafficLights:
                    if trafficLights[i//2][j][0] == 1: # horizontal
                        pygame.draw.line(screen, (255,255,255), (0+j*30,5+(i//2)*30), (10+j*30,5+(i//2)*30), trafficLightWidth)
                        if trafficLights[i//2][j][1] == 1:
                            pygame.draw.line(screen, (255,255,255), (5+j*30,10+(i//2)*30), (5+j*30,15+(i//2)*30), trafficLightWidth)
                        if trafficLights[i//2][j][2] == 1:
                            pygame.draw.line(screen, (255,255,255), (5+j*30,0+(i//2)*30), (5+j*30,-5+(i//2)*30), trafficLightWidth)
                    else:
                        pygame.draw.line(screen, (255,255,255), (5+j*30,0+(i//2)*30), (5+j*30,10+(i//2)*30), trafficLightWidth)
                        if trafficLights[i//2][j][1] == 1:
                            pygame.draw.line(screen, (255,255,255), (10+j*30,5+(i//2)*30), (15+j*30,5+(i//2)*30), trafficLightWidth)
                        if trafficLights[i//2][j][2] == 1:
                            pygame.draw.line(screen, (255,255,255), (0+j*30,5+(i//2)*30), (-5+j*30,5+(i//2)*30), trafficLightWidth)

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(1000)  # limits FPS to 60

    pygame.quit()

def pureComputation():
    global running
    while running:
        logic()

#pygameAnimation()

while True:
    initialize()
    #print(f'''{streetStates[1][1].cars} \n{streetStates[1][1].carIndeces}\n{activeCars}''')
    if graphicMode:
        pygameAnimation()
    else:
        pureComputation()
    print("----------------New Iteration--------------------")
    