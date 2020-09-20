
from queue import Queue, PriorityQueue
DEBUG = 1
def A_star_Traversal(
    #add your parameters 
):
    l = []

    return l

def UCS_Traversal(start_point, cost, goals):
    #print(cost)
    sols = []
    visited = set()
    queue = PriorityQueue()
    queue.put((0, start_point))
    number_of_nodes = len(cost[0]) - 1
    while queue:
        cost, node = queue.get()
        if node not in visited:
            visited.add(node)
            path = [node]
            if node in goals:
                sols.append([cost,path])

            for i in range(1,number_of_nodes+1):
                print(cost[node][i])
                if cost[node][i] != -1 and i not in visited:
                    path.append(i)
                    total_cost = cost + cost[node][i]
                    queue.put((total_cost,i))
    l = min(sols, key = lambda x : x[0])[1]
    print(l)
    return l
def findMinVertex(stack):
    min = 0
    for i in range(1,len(stack)):
        if stack[i] < stack[min]:
            min = i
    return stack.pop(min)
'''        
def DFS_Traversal(initial_state,final_states,cost):
    stack = []
    visited = set()
    stack.append(initial_state)
    path = []
    found = 0
    while len(stack):
        parent = findMinVertex(stack)
        if parent in visited:
            continue
        if parent in final_states:
            found = parent
            break
        visited.add(parent)
        children = [i for i in range(1,len(cost)) if cost[parent][i] > 0 ]
        if DEBUG:
            print("children",children)
        for child in children:
            stack.append(child)
'''       
              
    


def addToQueue(stack, vertex):
    for index, ele in enumerate(stack):
        if vertex > ele:
            stack.insert(index,vertex)     
def recurDFS(start, visited,cost,path,goals):
    stack = []
    stack.append(start)
    while len(stack):
        node = findMinVertex(stack)
        if not visited[node]:
            path.append(node)
            visited[node] = True   
        i = 1
        if node in goals:
            return 1
        while i < len(cost):
            if cost[node][i] > 0 and not visited[cost[node][i]]:
                if DEBUG:
                    print(node,i)
                #addToQueue(stack,i)
                stack.append(i)

            i+=1
    return -1
'''
def DFS_Traversal(initial_state,final_states,cost):
    visited = [False]*(len(cost))
    path = []
    ans = 0
    for i in range(1, len(cost)):
        
        if(not visited[i]):
            ans = recurDFS(i, visited,cost,path,final_states)
        if DEBUG:
            print(visited)    
        if ans:
            break
    if DEBUG:
        print(path,ans)    
    return path    
    

    path_cost = 0
    frontier = PriorityQueue()
    frontier.put((0, start_point)) 
    explored_set = []

    l = []
    while(1):
        if frontier.empty():
            break
        ucs_w, current_node = frontier.get()
        explored.append(current_node)  
        if current_node in goals:
            return
        for node in cost[current_node]:
            if node not in explored:
                frontier.put((
                    ucs_w + ucs_weight(current_node, node, weights),
                    node
                ))
    return l
'''
def DFS_Traversal(initial_state,final_states,cost):
    path = []
    stack = [initial_state]
    visited = set()

    while len(stack):
        if DEBUG:
            print("Stack values = ",stack)
        current_node = stack.pop()
        if current_node not in visited:
            visited.add(current_node)
            path.append(current_node)

        if current_node in final_states:
            break

        for neighbour in range(len(cost)-1,1,-1):
            if neighbour not in visited and cost[current_node][neighbour] > 0:
                if DEBUG:
                    print("Being added to stack = ",neighbour)
                stack.append(neighbour)
    if DEBUG:
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

    t1 = DFS_Traversal(start_point,goals,cost)
    
    #t2 = UCS_Traversal(start_point, cost, goals)

    t3 = A_star_Traversal(
    #send whatever parameters you require 
)
    #print(t2)
    l.append(t1)
    #l.append(t2)
    l.append(t3)
    return l

