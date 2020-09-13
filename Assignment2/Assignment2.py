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

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''

DFS_flag = 0
UCS_dict = {}
def DFS_recursive(n,cost,v,visited,goals):
    global DFS_flag
    if v not in visited and DFS_flag == 0:
        visited.append(v)
        if v in goals:
            DFS_flag = 1
            return

        for i in range(1,n):
            if DFS_flag != 0:
                return
            elif cost[v][i] != -1 and cost[v][i] != 0:
                DFS_recursive(n,cost,i,visited,goals)
            
def DFS_Traversal(cost,start_point,goals):
    n = len(cost)
    visited = []
    DFS_recursive(n,cost,start_point,visited,goals)
    return visited

def tri_Traversal(cost, heuristic, start_point, goals):
    l = []
    # t1 <= DFS_Traversal
    # t2 <= UCS_Traversal
    # t3 <= A_star_Traversal
    #t1 = DFS_Traversal(cost,start_point,goals)
    t2 = UCS_Traversal(cost,start_point,goals)
    #l.append(t1)
    l.append(t2)
    #l.append(t3)
    return l
