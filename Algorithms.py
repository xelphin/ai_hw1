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
      

def heuristic_msap(s,env):
    g_d_states = env.get_goal_states()[:]
    if s[1] == False:
        # didn't collect d1
        g_d_states += [env.d1]
    if s[2] == False:
        # didn't collect d2
        g_d_states += [env.d2]

    # find min manhatan distance
    s_row = env.to_row_col(s)[0]
    s_col = env.to_row_col(s)[1]
    min_dist = np.inf

    for g_d_state in g_d_states:
        g_row = env.to_row_col(g_d_state)[0]
        g_col = env.to_row_col(g_d_state)[1]
        dist = abs(s_row-g_row)+abs(s_col-g_col)
        min_dist = min(min_dist, dist)

    return min_dist


# ALGORITHMS

class BFSAgent():
    def __init__(self) -> None:
        self.env = None
        self.expandedCount = 0
        self.OPEN = []
        self.OPEN_INFO = []
        self.CLOSE = []

    class Path_Info():
        def __init__(self, actions = [], total_cost = 0, is_terminated = False) -> None:
            self.actions = actions
            self.total_cost = total_cost
            self.is_terminated = is_terminated

        def getActions(self):
            return self.actions[:]
        
        def getTotalCost(self):
            return self.total_cost
        
        def getIsTerminated(self):
            return self.is_terminated
        
    def resetAllAgentValues(self):
        self.env = None
        self.expandedCount = 0
        self.OPEN = []
        self.OPEN_INFO = []
        self.CLOSE = []

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.resetAllAgentValues()
        self.env = env
        self.env.reset()

        # node <- make_node(problem.init_state, null)
        state = self.env.get_initial_state()

        # if problem.goal(node.state) then return solution(node)
        if self.env.is_final_state(state):
            return ([],0,0)
        
        # OPEN <- {node}
        self.OPEN.append(state)
        self.OPEN_INFO.append(self.Path_Info([],0, False))
        # CLOSE <- {}

        # while OPEN is not empty do:
        while len(self.OPEN) != 0:
            # node <- OPEN.pop()
            state = self.OPEN.pop(0)
            state_info = self.OPEN_INFO.pop(0)
            # CLOSE.add(node.state)
            self.CLOSE.append(state)

            self.expandedCount += 1
            # print(f"{self.expandedCount} Expanding: {state}")
            if state_info.getIsTerminated():
                    continue # don't expand from terminated (old state)
            
            for action, (new_state, new_cost, new_terminated) in env.succ(state).items():
                # child <- make_node(s, node)
                env.reset()
                env.set_state(state) # now on parent state
                new_state, cost, terminated = self.env.step(action)
                new_state_path_info = self.Path_Info(state_info.getActions()+ [action], state_info.getTotalCost()+cost, terminated)

                # if child.state is not in CLOSE and child is not in OPEN:
                if new_state not in self.CLOSE and new_state not in self.OPEN:
                    # if problem.goal(child.state) then return solution(child)
                    if self.env.is_final_state(new_state):
                        return (new_state_path_info.getActions(), new_state_path_info.getTotalCost(), self.expandedCount)
                    # OPEN.insert(child)
                    self.OPEN.append(new_state)
                    self.OPEN_INFO.append(new_state_path_info)

        return [], 0,0



class WeightedAStarAgent():
    def __init__(self) -> None:
        self.env = None
        self.expandedCount = 0
        self.OPEN = heapdict.heapdict()
        self.CLOSED = heapdict.heapdict()

    class Node():
        def __init__(self, state, parentState, g_val, f_val, actions,  totalCost, is_terminated=False) -> None:
            self.state = state
            self.parentState = parentState
            self.g_val = g_val
            self.f_val = f_val
            self.actions = actions
            self.totalCost = totalCost
            self.is_terminated = is_terminated

        def get_state(self):
            return self.state
        
        def get_actions(self):
            return self.actions     

        def get_totalCost(self):
            return self.totalCost  

        def get_gVal(self):
            return self.g_val
        
        def get_fVal(self):
            return self.f_val
        
        def get_isTerminated(self):
            return self.is_terminated
        
    def nodeIsInHeapDict(self, state, hd):
        states_in_hd = [node.get_state() for node, f_val in hd.items()]
        return (state in states_in_hd)
    
    def getNodeInHeapDictUsingState(self, state, hd):
        for node, f_val in hd.items():
            if node.get_state() == state:
                return node
        return None
    
    def removeNodeFromHeapDict(self, node, hd):
        for k,v in hd.items():
            if k==node:
                hd.pop(k)
                return

    def resetAllAgentValues(self):
        self.env = None
        self.expandedCount = 0
        self.OPEN = heapdict.heapdict()
        self.CLOSED = heapdict.heapdict()

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.resetAllAgentValues()
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()
        state_h = heuristic_msap(state, env)
        
        # OPEN <- make_node(P.start, NIL, h(P.start))
        state_node = self.Node(state, None, 0, h_weight*state_h, [], 0, False)
        self.OPEN[state_node] = (h_weight*state_h, 0)

        # while OPEN != {}
        while len(self.OPEN) != 0:
            # n <- OPEN.pop_min
            state_node, state_f_val = self.OPEN.popitem()
            # CLOSED <- CLOSED + {n}
            self.CLOSED[state_node] = (state_node.get_fVal(), state_node.get_state()[0])

            # if P.goal_test(n)
            if self.env.is_final_state(state_node.get_state()):
                # return path (n)
                return (state_node.get_actions(), state_node.get_totalCost(), self.expandedCount)
            
            self.expandedCount += 1
            # print(f"{self.expandedCount} Expanding: {state_node.get_state()} f_val: {state_node.get_fVal()}")
            if state_node.get_isTerminated():
                # print(f"stopped expansion because terminated.")
                continue # don't expand from terminated

            # for s in P.SUCC(n)
            for action in range(4):
                env.reset()
                env.set_state(state_node.get_state()) # now on parent state
                new_state, cost, terminated = self.env.step(action)

                # new_g <-n.g() + P.COST(n.s,s)
                new_g = state_node.get_gVal()+cost
                # new_f <-new_g + h(s)
                new_f = (1-h_weight)*new_g + h_weight*heuristic_msap(new_state, env)
                new_state_node = self.Node(new_state, state, new_g, new_f, state_node.get_actions()+[action], state_node.get_totalCost()+cost, terminated)
                
                # if s not in OPEN+CLOSED
                if not self.nodeIsInHeapDict(new_state, self.OPEN) and not self.nodeIsInHeapDict(new_state, self.CLOSED):
                    # n' <- make_node(s,n,new_g, new_f)
                    # [new_state_node]
                    # OPEN.insert(n')
                    self.OPEN[new_state_node] = (new_f, new_state)
                    # print(f"state {new_state_node.get_state()} not in OPEN+CLOSED, adding a new state. f_val: {new_state_node.get_fVal()} ( g_val: {new_state_node.get_gVal()} h_val: {heuristic_msap(new_state, env)} )  terminated: {new_state_node.get_isTerminated()}")

                # else if s is in OPEN
                elif self.nodeIsInHeapDict(new_state, self.OPEN):
                    # n_curr <- node in OPEN with state s
                    n_curr = self.getNodeInHeapDictUsingState(new_state, self.OPEN)
                    # if new_f < n_curr.f()
                    if new_f < n_curr.get_fVal():
                        # n_curr <- update_node(s,n,new_g,new_f)
                        # OPEN.update_key(n_curr)
                        self.removeNodeFromHeapDict(n_curr, self.OPEN)
                        self.OPEN[new_state_node] = (new_f, new_state)
                    #     print(f"state {new_state_node.get_state()} in OPEN, updating the state. f_val: {new_state_node.get_fVal()}")
                    # else:
                    #     print(f"state {new_state_node.get_state()} in OPEN, IGNORING because f_val: {new_state_node.get_fVal()} didn't improve old_f_val {n_curr.get_fVal()}")

                # else if s is in CLOSED
                else:
                    # n_curr <- node in CLOSED with state s
                    n_curr = self.getNodeInHeapDictUsingState(new_state, self.CLOSED)
                    # if new_f < n_curr.f()
                    if new_f < n_curr.get_fVal():
                        # n_curr <- update_node(s,n,new_g,new_f)
                        # OPEN.insert(n_curr)
                        self.OPEN[new_state_node] = (new_f, new_state)
                        # CLOSED.remove(n_curr)
                        self.removeNodeFromHeapDict(n_curr, self.CLOSED)
                    #     print(f"state {new_state_node.get_state()} in CLOSED, putting back in OPEN with updated value. f_val: {new_state_node.get_fVal()}")
                    # else:
                    #     print(f"state {new_state_node.get_state()} in CLOSED, IGNORING because f_val: {new_state_node.get_fVal()} didn't improve old_f_val {n_curr.get_fVal()}")

                
            #  print(f"OPEN is now: {[(open_state_node.get_state(), open_state_node.get_fVal()) for open_state_node in self.OPEN]}")
        return ([],0,0)



class AStarEpsilonAgent():
    def __init__(self) -> None:
                self.env = None
                self.OPEN = heapdict.heapdict() # for f value
                self.CLOSED = heapdict.heapdict() # for comfort
                self.expandedCount = 0
                self.FOCAL = heapdict.heapdict() # for g(v)
    class Node():
            def __init__(self, state, parentState, g_val, f_val, actions,  totalCost, is_terminated=False) -> None:
                    self.state = state
                    self.parentState = parentState
                    self.g_val = g_val
                    self.f_val = f_val
                    self.actions = actions
                    self.totalCost = totalCost
                    self.is_terminated = is_terminated

            def get_state(self):
                    return self.state
                
            def get_actions(self):
                    return self.actions     

            def get_totalCost(self):
                    return self.totalCost  

            def get_gVal(self):
                    return self.g_val
                
            def get_fVal(self):
                    return self.f_val
                
            def get_isTerminated(self):
                    return self.is_terminated
                
    def nodeIsInHeapDict(self, state, hd):
        states_in_hd = [node.get_state() for node, f_val in hd.items()]
        return (state in states_in_hd)
    
    def getNodeInHeapDictUsingState(self, state, hd):
        for node, f_val in hd.items():
            if node.get_state() == state:
                return node
        return None
    
    def removeNodeFromHeapDict(self, node, hd):
        for k,v in hd.items():
            if k==node:
                hd.pop(k)
                return

    def resetAllAgentValues(self):
        self.env = None
        self.expandedCount = 0
        self.OPEN = heapdict.heapdict()
        self.CLOSED = heapdict.heapdict()
        self.FOCAL = heapdict.heapdict()
    
    def printnode(self, node):
         print("({},{},{}) - F(v)={}, g(v)={}\n".format(node.get_state()[0],node.get_state()[1],node.get_state()[2], node.get_fVal(), node.get_gVal()))

    def updateFocal(self, newF, trim, epsilon):
        # newf - new value of f will affect who enters focal list
        # trim - true - a lesser min was entered and the focal list will need to be trimmed. 
        # - false - the lowest f was removed so we need to add bigger f nodes from open
        # epsilon - provided by the user
        
        #if len(self.FOCAL)==0:
                #print("\nparams: newf:{}, trim:{}, epsilon:{}\n".format({newF}, {trim},{epsilon}))
                #for node, f_val in self.OPEN.items():
                    #print("({},{},{}) - F(v)={}\n".format(node.get_state()[0],node.get_state()[1],node.get_state()[2], node.get_fVal()))
         
        if trim: # if lower f was found:
            for node, f_val in self.FOCAL.items():
                if node.get_fVal() > newF*epsilon:
                    n_curr = self.getNodeInHeapDictUsingState(node, self.FOCAL)
                    self.removeNodeFromHeapDict(n_curr, self.FOCAL)
        
        else: # lowest f was removed => new larger f is now the min => open can have new focal-worthy nodes
            states_in_focal = [node.get_state() for node, f_val in self.FOCAL.items()]
            for node, f_val in self.OPEN.items():
                if not node.get_state() in states_in_focal:
                    if node.get_fVal() <= newF*epsilon:
                        self.FOCAL[node] = (node.get_gVal(), node.get_state()[0])
                      
    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        #recieve enviroment, epsilon, return (actions, cost, expanded)
        nepsilon=epsilon+1
        self.resetAllAgentValues()
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()
        state_h = heuristic_msap(state, env)

        state_node = self.Node(state, None, 0,state_h, [], 0, False)
        self.OPEN[state_node] = (state_h, 0) #f(v), index
        self.FOCAL[state_node] = (0, 0) # g(v), index

        while len(self.OPEN) != 0:
            if len(self.FOCAL)==0:
                print("\noof\n")
            state_node, state_f_val = self.FOCAL.popitem() # because we need minimum g(v) 
            #self.CLOSED[state_node] = (state_node.get_fVal(), state_node.get_state()[0])
            #F_limit = state_node.get_fVal()*epsilon

            self.CLOSED[state_node] = (state_node.get_fVal(), state_node.get_state()[0])

            nodet = self.getNodeInHeapDictUsingState(state_node.get_state(),self.OPEN)
            self.removeNodeFromHeapDict(nodet, self.OPEN)

            if self.env.is_final_state(state_node.get_state()):
                return (state_node.get_actions(), state_node.get_totalCost(), self.expandedCount)
            
            self.expandedCount += 1    
            if state_node.get_isTerminated():
                #TODO - Compare the f(v) of the hole to the minimal f(v) of the open list (not necessarily the same)
                # if it is the same - we need to find the next min f - we enlarge the min, 
                # so everything in focal needs to stay, and there could be more 
                # who should join focal

                if(len(self.OPEN) == 0):
                    break # no solution
                minf_node, minf_val = self.OPEN.peekitem()
                if state_node.get_fVal()< minf_node.get_fVal():
                    self.updateFocal(minf_node.get_fVal(), False, nepsilon)

                continue # don't expand from terminated

            for action in range(4):
                env.reset()
                env.set_state(state_node.get_state()) # now on parent state
                new_state, cost, terminated = self.env.step(action)

                new_g = state_node.get_gVal()+cost
                new_f = new_g + heuristic_msap(new_state, env)
                new_state_node = self.Node(new_state, state, new_g, new_f, state_node.get_actions()+[action], state_node.get_totalCost()+cost, terminated)

                # if s not in OPEN+CLOSED - new node
                if not self.nodeIsInHeapDict(new_state, self.OPEN) and not self.nodeIsInHeapDict(new_state, self.CLOSED):
                    self.OPEN[new_state_node] = (new_f, new_state_node.get_state()[0])
                    minf_node, minf_val = self.OPEN.peekitem()
                    if new_state_node.get_fVal()< minf_node.get_fVal()*nepsilon:
                        self.FOCAL[new_state_node] = (new_g,new_state_node.get_state()[0])
                        if new_state_node.get_fVal()< minf_node.get_fVal():
                            self.updateFocal(new_state_node.get_fVal(), True, nepsilon) # better f_val means trim focal
                    
                    


                # else if s is in OPEN
                elif self.nodeIsInHeapDict(new_state, self.OPEN):
                    

                    n_curr = self.getNodeInHeapDictUsingState(new_state, self.OPEN)
                    n_currf = self.getNodeInHeapDictUsingState(new_state, self.FOCAL)

                    if new_f < n_curr.get_fVal():
                        self.removeNodeFromHeapDict(n_curr, self.OPEN)
                        self.OPEN[new_state_node] = (new_f, new_state_node.get_state()[0])



                        if n_currf!=None:
                            self.removeNodeFromHeapDict(n_currf, self.FOCAL)

                        minf_node, minf_val = self.OPEN.peekitem()
                        if new_state_node.get_fVal()< minf_node.get_fVal()*nepsilon:
                            self.FOCAL[new_state_node] = (new_g,new_state_node.get_state()[0])
                            if new_state_node.get_fVal()< minf_node.get_fVal():
                                self.updateFocal(new_state_node.get_fVal(), True, nepsilon) # better f_val means trim focal

                # else if s is in CLOSED
                else:

                    n_curr = self.getNodeInHeapDictUsingState(new_state, self.CLOSED)

                    if new_f < n_curr.get_fVal():

                        self.OPEN[new_state_node] = (new_f, new_state_node.get_state()[0])

                        self.removeNodeFromHeapDict(n_curr, self.CLOSED)

                        minf_node, minf_val = self.OPEN.peekitem()
                        if new_state_node.get_fVal()< minf_node.get_fVal()*nepsilon:
                            self.FOCAL[new_state_node] = (new_g,new_state_node.get_state()[0])
                            if new_state_node.get_fVal()< minf_node.get_fVal():
                                self.updateFocal(new_state_node.get_fVal(), True, nepsilon) # better f_val means trim focal
            
            minf_node, minf_val = self.OPEN.peekitem()
            
            if state_node.get_fVal()< minf_node.get_fVal():
                # we finished with the min f state so someone else needs to take the throne
                #if(len(self.FOCAL)==0):
                #    self.FOCAL[minf_node] = (minf_node.get_gVal(),minf_node.get_state()[0])
                #print("\n just checked:({},{},{}) - F(v)={}\n".format(state_node.get_state()[0],state_node.get_state()[1],state_node.get_state()[2], state_node.get_fVal()))
                #print("\n new min :({},{},{}) - F(v)={}\n".format(minf_node.get_state()[0],minf_node.get_state()[1],minf_node.get_state()[2], minf_node.get_fVal()))
                self.updateFocal(minf_node.get_fVal(), False, nepsilon) # we finished with the state 
          
            '''if epsilon==0.1:
                print("\n just checked: ")
                self.printnode(state_node)    
                print("\n minf is: ")
                self.printnode(minf_node) 
                
                print("\nOPEN:\n")
                for node2, f_val in self.OPEN.items():
                    self.printnode(node2)  
                        #print("({},{},{}) - F(v)={}\n".format(node2.get_state()[0],node2.get_state()[1],node2.get_state()[2], node2.get_fVal()))
                print("\nFOCAL\n")
                for node3, f_val in self.FOCAL.items():
                        self.printnode(node3)              
                print("-------------------------------------------------------------")'''
        return ([],0,0)


            
                



                
            







