from collections import defaultdict 
import numpy as np
import pandas as pd
import AI_BayesianNetwork_V4_convertedIntoClassAndFunctionForAI_FindALlPaths as BN
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import warnings

class GrapH:
    # example of adjacency list (or rather map)
    # adjacency_list = {
    # 'A': [('B', 1), ('C', 3), ('D', 7)],
    # 'B': [('D', 5)],
    # 'C': [('D', 12)]
    # }

    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    # heuristic function with equal values for all nodes
    def h(self, n):
        H = {
#            'Los Angeles': 1,
#            'Phoenix': 1,
#            'Houston': 1,
#            'Chicago': 1,
#            'New York': 1
        #'Boston''LA'
#            
             'Sacramento': 1,
             'Minneapolis': 1,
             'Kansas City': 1,
             'Houston': 1,
             'Washington': 1
        }

        return H[n]

    def a_star_algorithm(self, start_node, stop_node):
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = set([start_node])
        closed_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v;

            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()
                print('\n\n***********************************************************************')
                print('************************* BEST PATH - A STAR **************************')
                print('***********************************************************************','\n')
                print('Best Path found using A*: {}'.format(reconst_path),'\n')
                
                ### Calculation of Distance and Time for the path showcased by A*
                
                totalDistanceOfPath = 0
                ### ASSIGNING DISTANCE TO EACH PATH
                for indexInPathWithCityName in range(len(reconst_path)-1):
                    ## Pick the consecutive pair of cities to find distance between them
    #                print(pathWithCityName[indexInPathWithCityName])       # 1st City
    #                print(pathWithCityName[indexInPathWithCityName + 1])   # 2nd City
                    ## Finding distance between the 2 cities using the distance Matrix csv
    #                print(distanceMatrixBetweenCities.loc[pathWithCityName[indexInPathWithCityName],pathWithCityName[indexInPathWithCityName + 1]])
                    
                    totalDistanceOfPath +=  distanceMatrixBetweenCities.loc[reconst_path[indexInPathWithCityName],reconst_path[indexInPathWithCityName + 1]]
                
                ### Assuming average speed of 60mph
                timeTakenToTravelOnPath = 0
                timeTakenToTravelOnPath = totalDistanceOfPath/60
                
                # timeWasted derived from AI_BayesianNetworkPrediction
#                timeWasted = BN.AI_BayesianNetwork_predictWastedTimepathWithCityName(pathWithCityName,departureTime, departureWeekDay)
    #            timeWasted = BN.AI_BayesianNetwork_predictWastedTimepathWithCityName(pathWithCityName)
                
                print('--> Travel time without considering Traffic Jam: ', round(timeTakenToTravelOnPath,2),'hours')
                timeWasted = BN.AI_BayesianNetwork_predictWastedTimepathWithCityName(reconst_path,departureTime, departureWeekDay)
                print('--> Total Distance to Cover                    :',totalDistanceOfPath,'miles')
                print('--> Travel time considering Traffic Jam: ', round(timeTakenToTravelOnPath + timeWasted,2),'hours\n')
                
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None
 

welcomeMessage = '''

                                Welcome!
      
      This system will predict the shortest path (time based) between 5 major cities of USA.
      
      Please enter the source and destination cities from the following list to get started:
            -> 'Sacramento'
            -> 'Minneapolis'
            -> 'Kansas City'
            -> 'Houston'
            -> 'Washington'
          '''
          
'''
            -> 'Los Angeles'
            -> 'Phoenix'
            -> 'Houston'
            -> 'New York'
'''
print(welcomeMessage)

source_city = input('Enter the source city: ')
destination_city = input('Enter the destination city: ') 
departureTime = input("Enter the time at which you'll leave (HH:MM): ")
departureWeekDay = input("Enter the WeekDay on which you'll leave: ")

#below distances are in miles
'''
adjacency_list = {
    'Los Angeles': [('Phoenix', 373), ('Houston', 3), ('New York', 7)],
    
    'Phoenix': [('Los Angeles', 1), ('Houston', 3), ('New York', 7)],
    'Houston': [('Phoenix', 1), ('Los Angeles', 3), ('New York', 7)],
    'New York': [('Phoenix', 1), ('Los Angeles', 3), ('Houston', 7)],
}'''
adjacency_list = {
    'Sacramento': [('Minneapolis', 1670), ('Kansas City', 1713), ('Houston', 1932)],
    'Minneapolis': [('Sacramento', 1930), ('Kansas City', 439), ('Washington', 1113)],
    'Kansas City': [('Sacramento', 1713), ('Minneapolis', 439), ('Houston', 746),('Washington', 1061)],
    'Houston': [('Sacramento', 1932), ('Kansas City', 746), ('Washington', 1410)],
    'Washington': [('Minneapolis', 1113), ('Kansas City', 1061), ('Houston', 1410)],
}
graph1 = GrapH(adjacency_list)
print('')
#graph1.a_star_algorithm(source_city, destination_city)
#pathWithCityName = ['Minneapolis','Kansas City', 'Houston']
#timeWasted = BN.AI_BayesianNetwork_predictWastedTimepathWithCityName(pathWithCityName)
#
#print(timeWasted)
#This class represents a directed graph  
# using adjacency list representation 


#source_city = 'Washington'
#destination_city = 'Sacramento'
#def possiblePathsAndPredictedPath(source_city, destination_city):
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################


class Graph: 
   
    def __init__(self,vertices): 
        #No. of vertices 
        self.V= vertices  
          
        # default dictionary to store graph 
        self.graph = defaultdict(list)  
   
    # function to add an edge to graph 
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
   
    '''A recursive function to print all paths from 'u' to 'd'. 
    visited[] keeps track of vertices in current path. 
    path[] stores actual vertices and path_index is current 
    index in path[]'''
    def printAllPathsUtil(self, u, d, visited, path): 
        
#        dictOfMappingCitieswithIndexNumber = {0:'Sacramento', 1: 'Minneapolis', 2:'Kansas City', 3: 'Houston', 4:'Washington'}

        # Mark the current node as visited and store in path 
        visited[u]= True
        path.append(u) 
  
        # If current vertex is same as destination, then print 
        # current path[] 
        if u ==d: 
#            print (path) 
            pathWithCityName = []
#            countOfNumberOfPossiblePaths += 1
            for city in path: # path is the same list as pathWithCityName..but contains digits instead of city names 
#                print(dictOfMappingCitieswithIndexNumber[city])
                pathWithCityName.append(dictOfMappingCitieswithIndexNumber[city])
            print('*************************************************************')
#            print('******************** POSSIBLE PATH',countOfNumberOfPossiblePaths, '************************')
            print('******************** POSSIBLE PATH **************************')
            print('*************************************************************')
            print()
            print(pathWithCityName)
            print()
            
            totalDistanceOfPath = 0
            ### ASSIGNING DISTANCE TO EACH PATH
            for indexInPathWithCityName in range(len(pathWithCityName)-1):
                ## Pick the consecutive pair of cities to find distance between them
#                print(pathWithCityName[indexInPathWithCityName])       # 1st City
#                print(pathWithCityName[indexInPathWithCityName + 1])   # 2nd City
                ## Finding distance between the 2 cities using the distance Matrix csv
#                print(distanceMatrixBetweenCities.loc[pathWithCityName[indexInPathWithCityName],pathWithCityName[indexInPathWithCityName + 1]])
                
                totalDistanceOfPath +=  distanceMatrixBetweenCities.loc[pathWithCityName[indexInPathWithCityName],pathWithCityName[indexInPathWithCityName + 1]]
            print('--> Total Distance to Cover                    :',totalDistanceOfPath,'miles')
            
            ### Assuming average speed of 60mph
            timeTakenToTravelOnPath = 0
            timeTakenToTravelOnPath = totalDistanceOfPath/60
            
            # timeWasted derived from AI_BayesianNetworkPrediction
            timeWasted = BN.AI_BayesianNetwork_predictWastedTimepathWithCityName(pathWithCityName,departureTime, departureWeekDay)
#            timeWasted = BN.AI_BayesianNetwork_predictWastedTimepathWithCityName(pathWithCityName)
            
            print('--> Travel time without considering Traffic Jam: ', round(timeTakenToTravelOnPath,2),'hours')
            print('--> Travel time considering Traffic Jam: ', round(timeTakenToTravelOnPath + timeWasted,2),'hours\n')
            print()
            ## Combine the path and the time in a set....This will help in sorting based on time and output the shortest path
#            listOfPathAndTime = []
#            listOfPathAndTime = set()
#            listOfPathAndTime.add(listOfPathAndTime)
#            listOfPathAndTime.add(timeTakenToTravelOnPath)
            ## Had to round of the time as Dictionary keys can't be float
            listOfPathAndTime[round(timeTakenToTravelOnPath+timeWasted)] = pathWithCityName
            
            ## Distance and Time are stored
            listOfDistanceAndTime[round(timeTakenToTravelOnPath+timeWasted)] = totalDistanceOfPath
#            listOfPathAndTime[round(timeTakenToTravelOnPath)] = totalDistanceOfPath
            #The value of the above dictionary is printed below in the main program
#            listOfPathAndTime.append(timeTakenToTravelOnPath)
#            listOfPathAndTime.append(pathWithCityName)
#            print(set(listOfPathAndTime))
#            print(listOfPathAndTime.iteritems())
            
            
        else: 
            # If current vertex is not destination 
            #Recur for all the vertices adjacent to this vertex 
            for i in self.graph[u]: 
                if visited[i]==False: 
                    self.printAllPathsUtil(i, d, visited, path) 
                      
        # Remove current vertex from path[] and mark it as unvisited 
        path.pop() 
        visited[u]= False
   
   
    # Prints all paths from 's' to 'd' 
    def printAllPaths(self,s, d): 
  
        # Mark all the vertices as not visited 
        visited =[False]*(self.V) 
  
        # Create an array to store paths 
        path = [] 
  
        # Call the recursive helper function to print all paths 
        self.printAllPathsUtil(s, d,visited, path) 
   
   
   
# Create a graph given in the above diagram 
g = Graph(5) 

## PATHS from west to East US
g.addEdge(0, 1) 
g.addEdge(0, 2) 
g.addEdge(0, 3) 
g.addEdge(1, 2) 
g.addEdge(2, 3) 
g.addEdge(2, 4) 
g.addEdge(1, 4) 
g.addEdge(3, 4) 

## PATHS from East to West US
g.addEdge(1, 0) 
g.addEdge(2, 0) 
g.addEdge(3, 0) 
g.addEdge(2, 1) 
g.addEdge(3, 2) 
g.addEdge(4, 2) 
g.addEdge(4, 1) 
g.addEdge(4, 3) 


#g.add_edge('Sacramento', 'Minneapolis', weight=0.6)
#g.add_edge('Sacramento', 'Kansas City', weight=0.2)
#g.add_edge('Sacramento', 'Houston', weight=0.1)
#g.add_edge('Minneapolis', 'Kansas City', weight=0.7)
#g.add_edge('Kansas City', 'Houston', weight=0.9)
#g.add_edge('Kansas City', 'Washington', weight=0.3)
#g.add_edge('Minneapolis', 'Washington', weight=0.3)
#g.add_edge('Houston', 'Washington', weight=0.3)

#g.addEdge('Sacramento', 'Minneapolis')
#g.addEdge('Sacramento', 'Kansas City')
#g.addEdge('Sacramento', 'Houston')
#g.addEdge('Minneapolis', 'Kansas City')
#g.addEdge('Kansas City', 'Houston')
#g.addEdge('Kansas City', 'Washington')
#g.addEdge('Minneapolis', 'Washington')
#g.addEdge('Houston', 'Washington')
#   



## Stores and  Combine the path and the time in a dict....This will help in sorting based on time and output the shortest path
listOfPathAndTime = {}

## Stores and  Combine the Distance and the time in a dict....This will help in sorting based on time and output the distance
#Mainly helpful to show comparison that even though distance of a path is more, time is less
listOfDistanceAndTime = {}
#    countOfNumberOfPossiblePaths = 0

dictOfMappingCitieswithIndexNumber = {0:'Sacramento', 1: 'Minneapolis', 2:'Kansas City', 3: 'Houston', 4:'Washington'}

##### Conversion of user input (City Names) to Digits using the dictionary of City Names

#    key_list_CityNameMapping_dictOfMappingCitieswithIndexNumber = list(dictOfMappingCitieswithIndexNumber.keys()) 
#    val_list_CityNameMapping_dictOfMappingCitieswithIndexNumber = list(dictOfMappingCitieswithIndexNumber.values()) 
  
#print(key_list_CityNameMapping_dictOfMappingCitieswithIndexNumber[val_list_CityNameMapping_dictOfMappingCitieswithIndexNumber.index('Sacramento')]) 
#print(key_list_CityNameMapping_dictOfMappingCitieswithIndexNumber[val_list_CityNameMapping_dictOfMappingCitieswithIndexNumber.index('Washington')]) 
  
# one-liner 
#***************When integrating, change below index values 'Kansas....to user defined values....specify the input var
s = list(dictOfMappingCitieswithIndexNumber.keys())[list(dictOfMappingCitieswithIndexNumber.values()).index(source_city)]
d = list(dictOfMappingCitieswithIndexNumber.keys())[list(dictOfMappingCitieswithIndexNumber.values()).index(destination_city)]

#s = list(dictOfMappingCitieswithIndexNumber.keys())[list(dictOfMappingCitieswithIndexNumber.values()).index('Washington')]
#d = list(dictOfMappingCitieswithIndexNumber.keys())[list(dictOfMappingCitieswithIndexNumber.values()).index('Sacramento')]

#s = list(dictOfMappingCitieswithIndexNumber.keys())[list(dictOfMappingCitieswithIndexNumber.values()).index('Sacramento')]
#d = list(dictOfMappingCitieswithIndexNumber.keys())[list(dictOfMappingCitieswithIndexNumber.values()).index('Washington')]
#

print('\n*************************************************************\n')
print ("Following are all different paths from ", dictOfMappingCitieswithIndexNumber[s], 'to',dictOfMappingCitieswithIndexNumber[d], ':\n') 
g.printAllPaths(s, d) 


###### Assign 'time taken' value to each of the above path, assuming avg speed of 60mph 
#distanceMatrixBetweenCities = np.empty((5,5,))
#distanceMatrixBetweenCities[:] = np.nan
#
#distanceMatrixBetweenCities[]
#print(distanceMatrixBetweenCities)

print()
distanceMatrixBetweenCities =  pd.read_csv('Dataset\\DistanceMatrixBetweenCities.csv')
#distanceMatrixBetweenCities.rename(index = {0:'Sacramento'})

warnings.filterwarnings("ignore")
distanceMatrixBetweenCities = distanceMatrixBetweenCities.set_index('City')
print('***********************************************************************')
print('*********************** DISTANCE MATRIX *******************************')
print('***********************************************************************','\n')
print(distanceMatrixBetweenCities)

### Assigning of distance between the different paths is done in printAllPathsUtil function above
#print(listOfPathAndTime)  #Before sorting
#print()
## sorting the dictionary based on the minimum time taken
listOfPathAndTime = sorted(listOfPathAndTime.items(), key = lambda x : x[0])
listOfDistanceAndTime = sorted(listOfDistanceAndTime.items(), key = lambda x : x[0])

### Print the actual shortest path using Bayesian
#print(listOfPathAndTime)
print('\n\n***********************************************************************')
print('******************* BEST PREDICTED PATH - BAYESIAN ********************')
print('***********************************************************************','\n')

print('The actual path that is shortest based on TIME (considering traffic jams): ')
print('\n--> ',listOfPathAndTime[0][1],'\n')  
print('--> Actual Travel Time: ',listOfPathAndTime[0][0],'hours')
print('--> Total travel distance: ',listOfDistanceAndTime[0][1],'miles')

#A star executed
graph1.a_star_algorithm(source_city, destination_city)


################################
################################
#PLOTTING OF GRAPH TO SHOWCASE THE PATHS BETWEEN CITIES
################################
################################


G=nx.Graph()
G.add_edge('Sacramento', 'Minneapolis', weight=0.6)
G.add_edge('Sacramento', 'Kansas City', weight=0.2)
G.add_edge('Sacramento', 'Houston', weight=0.1)
G.add_edge('Minneapolis', 'Kansas City', weight=0.7)
G.add_edge('Kansas City', 'Houston', weight=0.9)
G.add_edge('Kansas City', 'Washington', weight=0.3)
G.add_edge('Minneapolis', 'Washington', weight=0.3)
G.add_edge('Houston', 'Washington', weight=0.3)

nx.draw(G,with_labels=True )
plt.savefig("simple_path.png") # save as png
plt.show() # display
#source_city = 'Sacramento'
#destination_city = 'Washington'
#possiblePathsAndPredictedPath(source_city, destination_city)
#possiblePathsAndPredictedPath('Sacramento' , 'Washington')