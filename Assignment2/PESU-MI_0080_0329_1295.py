
from queue import Queue, PriorityQueue
DEBUG = 0

def A_star_Traversal(
    initial_state, final_states, cost, heuristic
): 
    path = []
    queue = PriorityQueue()
    visited = set()
    queue.put((0 + heuristic[initial_state], initial_state, [initial_state]))
    while not queue.empty():
        f_path_cost, current_node, path = queue.get()
        visited.add(current_node)
        if current_node in final_states:
            if DEBUG:
                print("[A STAR] ",path)
            return path       
        children = cost[current_node]
        for child in range(1, len(children)):
            if child not in visited and children[child] > 0:
                    queue.put((f_path_cost + children[child] + heuristic[child] - heuristic[current_node], child, path + [child]))
        if DEBUG:
            print("[A STAR] Current f_path cost ",f_path_cost)
            print("[A STAR] Current node ",current_node)
            print("[A STAR] Path to current node ",path) 
    return []

def UCS_Traversal(
   initial_state,final_states,cost
):
    path = []
    queue = PriorityQueue()
    visited = set()
    queue.put((0, initial_state, [initial_state]))
    while not queue.empty():
        path_cost, current_node, path = queue.get()
        visited.add(current_node)
        if current_node in final_states:
            return path      
        children = cost[current_node]
        for child in range(1, len(children)):
            if child not in visited and children[child] > 0:
                    queue.put((path_cost + children[child], child, path + [child]))
        if DEBUG:
            print("[UCS] Current path cost ",path_cost)
            print("[UCS] Current node ",current_node)
            print("[UCS] Path to current node ",path) 

    return []

def DFS_Traversal(
    initial_state,final_states,cost
):
    path = []
    stack = [initial_state]
    visited = set()

    while len(stack):
        current_node = stack.pop()
        if current_node not in visited:
            visited.add(current_node)
            path.append(current_node)

        if current_node in final_states:
            if DEBUG:
                print("[DFS] ",path)
            return path
        no_neighbour = 1
        for neighbour in range(len(cost)-1,0,-1):
            if neighbour not in visited and cost[current_node][neighbour] > 0:
                stack.append(neighbour)
                no_neighbour = 0
        if no_neighbour and len(path):
                stack.append(path[-1])
                children = [i for i in range(len(cost)) if i not in visited and cost[path[-1]][i] > 0]
                if len(children) == 0:
                    path.pop() 
        if DEBUG:        
            print("[DFS] Current value of stack ",stack) 
            print("[DFS] no_neighbour value ",no_neighbour)       
                            
    return []
'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''

def tri_traversal(cost, heuristic, start_point, goals):

    l = []
        
    t1 = DFS_Traversal(start_point, goals, cost)
    
    t2 = UCS_Traversal(start_point, goals, cost)

    t3 = A_star_Traversal(start_point, goals, cost, heuristic)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l

