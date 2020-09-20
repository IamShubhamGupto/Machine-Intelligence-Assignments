import copy
import Queue as Q
# graph dictonary contains all the adjacent nodes of each as key and value pair
# cost_dict contains cost of each edge traversal of (u,v)
# final_dict contains all the possible paths from start node s to goal node g with total cost
class Graph():
    def __init__(self):
        self.graph = dict()
        self.cost_dict=dict()
        self.final_dict=dict()

    # u and v are nodes: edge u-->v with cost c 
    def addEdge(self,u,v,c):
        
        if u not in self.graph:
            qu = Q.PriorityQueue()
            self.graph.update({u:qu})

        self.graph[u].put(v)
        self.cost_dict.update({(u,v):c})
        
    # Makes a list to keep track of visited nodes
    def tnode(self,n):
        self.visited = [False]*n


    def UCS_util(self,s,visited,path,goal,value):
        # Appending node to the current path 
        path.append(s)
        # Marking that node is visited 
        visited[s]=True
        
        # If goal node is reached save the path and return
        if goal==s:
            self.final_dict.update({tuple(path):value})
            return
        
        # Checking if the adjacent node is been visited and explore the new path if haven't
        for i in self.graph[s].queue:
            if visited[i]==False:
                # When new path is being explored add the cost of that path to cost of entire course traversal
                # Send a copy of path list to avoid sending it by reference
                self.UCS_util(i,copy.deepcopy(visited),copy.deepcopy(path),goal,value + self.cost_dict[s,i])

    def UCS(self, s,goal):
        self.visited[s] = True
        # List to hold all the nodes visited in path from start node to goal node 
        path=[s]
        
        for i in self.graph[s].queue:
            if self.visited[i] == False:
                # Make a variable to hold the cost of traversal
                value = self.cost_dict[s,i]
                self.UCS_util(i,copy.deepcopy(self.visited),copy.deepcopy(path),goal,value)

    # Display all the paths that is been discovered from start node to Goal node
    def all_paths(self):
        # Check if there is any path
        if bool(self.final_dict):
            print("All the paths: ")
            for i in self.final_dict:
                print "path: ",i,"cost: ",self.final_dict[i]
        else:
            print "No Path exist between start and goal node"

    # Find the most optimal path between start node to goal node
    def optimal_path(self):
        if bool(self.final_dict):
            print "best path: ",min(self.final_dict, key=self.final_dict.get)
        else:
            print "No Path exist between start and goal node"

g = Graph()
g.tnode(10)
g.addEdge(0, 1, 1); g.addEdge(0, 2, 1); g.addEdge(1, 3, 3);
g.addEdge(2, 5, 2); g.addEdge(3, 6, 4); g.addEdge(3, 5, 2);
g.addEdge(4, 6, 1); g.addEdge(4, 7, 5); g.addEdge(5, 4, 4);
g.addEdge(6, 7, 1); g.addEdge(5, 0, 3); g.addEdge(5, 8, 1);
g.addEdge(8, 4, 1); g.addEdge(8, 9, 3); g.addEdge(9, 7, 1);
g.UCS(0,7)
g.all_paths()
g.optimal_path()