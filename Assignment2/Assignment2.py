
from queue import Queue, PriorityQueue
DEBUG = 0
def A_star_Traversal(
    #add your parameters 
):
    l = []

    return l
def UCS_Traversal(start_point, cost, goals):   
    return []
def DFS_Traversal(initial_state,final_states,cost):
    path = []
    stack = [initial_state]
    visited = set()

    while len(stack):
        #print(lexicographical_vertex)
        #min_vertex = min(min(stack), lexicographical_vertex)    
        #print(min_vertex)
        current_node = stack.pop()
        if current_node not in visited:
            visited.add(current_node)
            path.append(current_node)

        if current_node in final_states:
            break
        no_neighbour = 1
        for neighbour in range(len(cost)-1,0,-1):
            #print(current_node)
            if neighbour not in visited and cost[current_node][neighbour] > 0:
                if DEBUG:
                    print("Being added to stack = ",neighbour)  
                stack.append(neighbour)
                no_neighbour = 0
        if no_neighbour:
            stack.append(path.pop())
            children = [i for i in range(len(cost)) if i not in visited and cost[stack[-1]][i] > 0]
            if len(children) > 0:
                path.append(stack[-1])
        #print(stack)        
                
    #if DEBUG:
    print(path)            
    return path
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
    
    adjMat = [[0 for i in range(len(cost[0]))] for j in range(len(cost))]
    for i in range(0,len(cost)):
        for j in range(len(cost[0])):
            adjMat[i][j] = 1 if cost[i][j] > 0 else 0
    #for ele in adjMat:
        #print(ele)  
    '''
    adjList = dict()
    for i in range(len(cost)):
        adjList[i] = [ j for j in range(len(cost[i])) if cost[i][j] > 0]
    print(adjList)  
    ''' 
    
    t1 = DFS_Traversal(start_point,goals,adjMat)
    
    t2 = UCS_Traversal(start_point, cost, goals)

    t3 = A_star_Traversal(
    #send whatever parameters you require 
)
    #print(t2)
    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l

