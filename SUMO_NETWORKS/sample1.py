#!/usr/bin/env python

import traci
from sumolib import checkBinary  # Checks for the binary in environ vars
import numpy as np
import os
import sys
import optparse
import random

import math
import sumolib
import networkx as nx
import xml.etree.ElementTree as ET

net = sumolib.net.readNet('sample1.net.xml')

graph = nx.DiGraph()

print("Available Nodes : ", end=" ")
for node in net.getNodes():
    print(node.getID(), end=" ")
print()
source_node = input("FROM : ")
dest_node = input("TO : ")

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


# contains TraCI control loop
def short_path():
    graph = nx.DiGraph()
    for edge in net.getEdges():
        graph.add_edge(edge.getFromNode().getID(),
                       edge.getToNode().getID(), weight=edge.getLength())
    start_node = "J9"  # Replace with your start node ID
    end_node = "J3"     # Replace with your end node ID
    shortest_path = nx.shortest_path(
        graph, source=start_node, target=end_node, weight='weight')

    edges = net.getEdges()

    # Find the edge connecting the two junctions
    print(shortest_path)
    everyEdges = []
    for i in range(len(shortest_path)-1):
        connecting_edge = None
        for edge in edges:
            from_junction = edge.getFromNode().getID()
            to_junction = edge.getToNode().getID()
            if from_junction == shortest_path[i] and to_junction == shortest_path[i+1]:
                connecting_edge = edge
                everyEdges.append(connecting_edge)
                break

    finalEdges = []
    for edge in everyEdges:
        xml_string = '' + str(edge)
        root = ET.fromstring(xml_string)
        edge_id = root.attrib['id']
        finalEdges.append(str(edge_id))

    return finalEdges


def findDistance(start, end):
    graph = nx.DiGraph()
    for edge in net.getEdges():
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        graph.add_edge(from_node, to_node, lanes=edge.getLanes())
    all_paths = list(nx.all_simple_paths(graph, start, end))
    minLenPath = None
    low = 1000000
    for path in all_paths:
        if len(path) < low:
            low = len(path)
            minLenPath = path
    dist = 0
    for i in range(len(minLenPath)-1):
        # junction_1_coords = traci.junction.getPosition(minLenPath[i])
        # junction_2_coords = traci.junction.getPosition(minLenPath[i+1])
        dist += graph[minLenPath[i]][minLenPath[i+1]]["lanes"][0].getLength()
    return dist


def ACO(start, end):
    graph = nx.DiGraph()
    for edge in net.getEdges():
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        graph.add_edge(from_node, to_node, lanes=edge.getLanes())

    # print(graph["J0"]["J1"]["lanes"][0].getLength())
    all_paths = list(nx.all_simple_paths(graph, start, end))

    maxCities = len(net.getNodes())
    max_int = np.iinfo(np.int64).min
    distances = np.zeros((maxCities, maxCities))
    for path in all_paths:
        for i in range(len(path)-1):
            # junction_1_coords = traci.junction.getPosition(path[i])
            # junction_2_coords = traci.junction.getPosition(path[i+1])
            dist = graph[path[i]][path[i+1]]["lanes"][0].getLength()
            distances[int(path[i][1:]), int(path[i+1][1:])] = dist
            distances[int(path[i+1][1:]), int(path[i][1:])] = dist
    # print(distances)

    for i in range(len(distances)):
        for j in range(len(distances[i])):
            if i != j and distances[i, j] == 0:
                distances[i, j] = findDistance(
                    "J"+str(i), "J"+str(j))
    # print(distances)
    graph = np.array(distances)

    # ACO parameters
    num_ants = 10
    num_iterations = 500
    pheromone_initial = 1.0
    alpha = 1.0  # Pheromone importance factor
    beta = 2.0   # Heuristic importance factor
    evaporation_rate = 0.5

    # Initialize pheromone matrix
    num_nodes = len(graph)
    pheromone_matrix = np.ones_like(graph) * pheromone_initial

    # ACO function to find shortest path

    def ant_colony_optimization(graph, pheromone_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, start_node, end_node):
        best_path = None
        best_distance = float('inf')

        for iteration in range(num_iterations):
            paths = []
            path_distances = []

            for ant in range(num_ants):
                visited = [start_node]
                current_node = start_node

                while current_node != end_node:
                    unvisited = [node for node in range(
                        num_nodes) if node not in visited]
                    probabilities = np.zeros(num_nodes)

                    for node in unvisited:
                        pheromone = pheromone_matrix[current_node, node]
                        # Avoid division by zero
                        heuristic = 1 / (graph[current_node, node] + 1e-10)
                        probabilities[node] = (
                            pheromone ** alpha) * (heuristic ** beta)

                    probabilities /= np.sum(probabilities)
                    next_node = np.random.choice(unvisited)

                    visited.append(next_node)
                    current_node = next_node

                path_distance = sum(graph[visited[i], visited[i + 1]]
                                    for i in range(len(visited) - 1))
                paths.append(visited)
                path_distances.append(path_distance)

                if path_distance < best_distance:
                    best_distance = path_distance
                    best_path = visited

            # Update pheromone levels
            pheromone_matrix *= (1 - evaporation_rate)
            for path, distance in zip(paths, path_distances):
                for i in range(len(path) - 1):
                    pheromone_matrix[path[i], path[i + 1]] += 1 / distance
        return best_path, best_distance

    # Run ACO algorithm to find shortest path between nodes 0 and 4
    shortest_path, shortest_distance = ant_colony_optimization(
        graph, pheromone_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, start_node=int(start[1:]), end_node=int(end[1:]))

    # Output shortest path and distance
    finalJuncitons = []
    for junction in shortest_path:
        finalJuncitons.append("J"+str(junction))

    edges = net.getEdges()

    graph = nx.DiGraph()
    for edge in net.getEdges():
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        graph.add_edge(from_node, to_node)

    veryFinalJunctions = []
    for i in range(len(finalJuncitons)-1):
        all_paths = list(nx.all_simple_paths(
            graph, finalJuncitons[i], finalJuncitons[i+1]))
        minLenPath = None
        low = 1000000
        for path in all_paths:
            if len(path) < low:
                low = len(path)
                minLenPath = path
        veryFinalJunctions.extend(minLenPath)
    veryFinalJunctions = list(dict.fromkeys(veryFinalJunctions))

    everyEdges = []
    for i in range(len(veryFinalJunctions)-1):
        connecting_edge = None
        for edge in edges:
            from_junction = edge.getFromNode().getID()
            to_junction = edge.getToNode().getID()
            if from_junction == veryFinalJunctions[i] and to_junction == veryFinalJunctions[i+1]:
                connecting_edge = edge
                everyEdges.append(connecting_edge)
                break

    finalEdges = []
    for edge in everyEdges:
        xml_string = '' + str(edge)
        root = ET.fromstring(xml_string)
        edge_id = root.attrib['id']
        finalEdges.append(str(edge_id))

    return [finalEdges, shortest_distance]


def run():
    routeInfo = ACO(start=source_node, end=dest_node)
    route = routeInfo[0]
    shortest_distance = routeInfo[1]
    print("Shortest Distance : ", shortest_distance)
    traci.route.add("trip", route)
    for i in range(0, 10):
        veh_name = "veh_" + str(i)
        traci.vehicle.add(veh_name, "trip")

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()


def randomRouteGenerator():
    routes = [["-E9", "-E8", "-E7", "-E6"],
              ["E25", "E26", "E27", "E28", "E29", "E30", "E31", "E32", "E33"],
              ["E0", "E1", "E2", "E3", "E4", "E5"],
              ["E10", "E11", "E12", "E13", "E14", "E15", "E16"],
              ["-E24", "-E23", "-E22", "-E21", "-E20", "-E19", "-E18", "-E17"]]
    randomIndex = random.randint(0, len(routes)-1)
    return routes[randomIndex]


# main entry point
if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "sample1.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    run()
