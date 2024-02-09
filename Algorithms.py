import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict


# DEBUGGING
import time
from IPython.display import clear_output
def print_solution(actions,env: DragonBallEnv) -> None:
    env.reset()
    total_cost = 0
    print(env.render())
    print(f"Timestep: {1}")
    print(f"State: {env.get_state()}")
    print(f"Action: {None}")
    print(f"Cost: {0}")
    time.sleep(1)

    for i, action in enumerate(actions):
      state, cost, terminated = env.step(action)
      total_cost += cost
      clear_output(wait=True)

      print(env.render())
      print(f"Timestep: {i + 2}")
      print(f"State: {state}")
      print(f"Action: {action}")
      print(f"Cost: {cost}")
      print(f"Total cost: {total_cost}")
      
      time.sleep(1)

      if terminated is True:
        break

# HELPER
      
class Path_Info():
    def __init__(self, actions = [], total_cost = 0, d1=False, d2=False) -> None:
        self.actions = actions
        self.total_cost = total_cost
        self.d1 = d1
        self.d2 = d2

    def addAction(self, action):
        self.actions.append(action)

    def addToTotalCost(self, total_cost):
        self.total_cost += total_cost

    def addDragonBall1(self):
        self.d1 = True
    
    def addDragonBall2(self):
        self.d2 = True

    def getActions(self):
        return self.actions[:]
    
    def getTotalCost(self):
        return self.total_cost
    
    def getDragonBall1(self):
        return self.d1
    
    def getDragonBall2(self):
        return self.d2

# ALGORITHMS

class BFSAgent():
    def __init__(self) -> None:
        self.env = None
        self.expandedCount = 0
        self.OPEN = []
        self.OPEN_INFO = []
        self.CLOSE = []

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()

        # node <- make_node(problem.init_state, null)
        state = self.env.get_initial_state()

        # if problem.goal(node.state) then return solution(node)
        if self.env.is_final_state(state):
            return ([],0,0)
        
        # OPEN <- {node}
        self.OPEN.append(state)
        self.OPEN_INFO.append(Path_Info([],0, self.env.d1, self.env.d2))
        # CLOSE <- {}

        # while OPEN is not empty do:
        while len(self.OPEN) != 0:
            # node <- OPEN.pop()
            state = self.OPEN.pop(0)
            state_info = self.OPEN_INFO.pop(0)
            # CLOSE.add(node.state)
            self.CLOSE.append(state)

            # loop for s in expand(node.state)
            self.expandedCount += 1
            print(f"{self.expandedCount} Expanding: state {state}. Where: actions = {state_info.getActions()}, cost = {state_info.getTotalCost()} dragonballs = {state_info.getDragonBall1()},{state_info.getDragonBall2()}")
            for action, successor in env.succ(state).items():
                # child <- make_node(s, node)
                env.set_state(state)
                # env.d1 = state_info.getDragonBall1() # DOESN'T WORK
                # env.d2 = state_info.getDragonBall2() # DOESN'T WORK
                new_state, cost, terminated = self.env.step(action)
                new_state_path_info = Path_Info(state_info.getActions()+ [action], state_info.getTotalCost()+cost, env.d1, env.d1)

                # if child.state is not in CLOSE and child is not in OPEN:
                if new_state not in self.CLOSE and new_state not in self.OPEN and not (terminated is True and self.env.is_final_state(new_state) is False):
                    # if problem.goal(child.state) then return solution(child)
                    if self.env.is_final_state(new_state):
                        print(f"Found Solution: {self.expandedCount}, state {new_state}. Where: actions = {new_state_path_info.getActions()}, cost = {new_state_path_info.getTotalCost()}")
                        print_solution(new_state_path_info.getActions(), env)
                        return (new_state_path_info.getActions(), new_state_path_info.getTotalCost(), self.expandedCount)
                    # OPEN.insert(child)
                    self.OPEN.append(new_state)
                    self.OPEN_INFO.append(new_state_path_info)


        return [], -1,-1



class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError