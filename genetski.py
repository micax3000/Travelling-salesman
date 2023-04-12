import bisect

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator
import random


# --------------------------------------------------------
def initialize_a_route(starting_point, length):
    route = list(range(1, length + 1))
    route.remove(starting_point)
    route = [starting_point] + route + [starting_point]
    return route


# izracunava distancu izmedju svih gradova
def fitness(member, adjacency_matrix):
    fitness = 0
    member_len = len(member)
    for i in range(0, member_len - 1):
        x = member[i]
        y = member[i + 1]
        fitness += adjacency_matrix[x - 1][y - 1]

    return fitness


# izracunava duzinu svih puteva u odredjenoj generaciji i sortira ih u opadajucem redosledu
# put sa najmanjom duzinom je na kraju liste
def population_fitness(population, adjacency_matrix):
    pop_fitness = {}
    for i in range(0, len(population)):
        pop_fitness[i] = fitness(population[i], adjacency_matrix)
    return sorted(pop_fitness.items(), key=operator.itemgetter(1), reverse=True)


def create_starting_population(size, route):
    population = []
    for _ in range(size):
        shuffled = route[1:-1]
        random.shuffle(shuffled)
        population.append([route[0]] + shuffled + [route[-1]])
    return population


# bira jedinke za ukrstanje i vraca njihove indekse
# indeksi jedinki se mogu pojaviti vise puta
def selection(popRanked, eliteSize):
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['probability'] = 100 * df.cum_sum / df.Fitness.sum()
    selectionResults = [gene[0] for gene in popRanked[len(popRanked) - eliteSize:]]
    for _ in range(len(popRanked) - eliteSize):
        pick = 100 * random.random()
        firstLarger = first_larger_number_index(df['probability'], pick)
        if firstLarger != -1:
            selectionResults.append(popRanked[firstLarger][0])
        # for j in range(len(popRanked)):
        #     if pick <= df.iat[j, 3]:
        #         selectionResults.append((popRanked[j][0]))
        #         break
    return selectionResults


def first_larger_number_index(lst, x):
    index = bisect.bisect_left(lst, x)
    if index == len(lst):
        return -1
    else:
        return index


# ukrstavanje dve jedinke
# dete uzima niz gena ordredjene duzine od prvog roditelja
# sve ostale gene do len(a)(ili len(b)) uzima od drugog roditelja sa tim da se geni ne smeju ponavljati
def breed(a, b):
    geneA = random.randrange(1,len(b)-2)
    geneB = random.randrange(1,len(b)-2)

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    part1 = a[startGene:endGene]
    child = [gene for gene in b if gene not in part1]
    child[startGene:startGene] = part1

    return child


# vrsi ukrstavanje populacije sa elitizmom
# ukrstanje po parovima
def breedPopulation(pool, elitism):
    bredPopulation = pool[:elitism]

    for i in range(0, len(pool) - elitism):
        child = breed(pool[i], pool[len(pool) - i - 1])
        bredPopulation.append(child)
    return bredPopulation


# mutiranje jedinki predstavlja zamenu mesta nasumicno izabranih gena
def mutate(chromosome, mutationRate):
    for indx in range(1, len(chromosome) - 1):
        if (random.random() < mutationRate):
            indx2 = random.randrange(1, len(chromosome) - 2)

            chromosome[indx],chromosome[indx2] = chromosome[indx2],chromosome[indx]

    return chromosome


# mutira celu populaciju
def mutatePopulation(matingPool, mutationRate, elitism):
    pop = matingPool[:elitism]
    for i in range(elitism, len(matingPool)):
        chromosome = matingPool[i]
        mutated = mutate(chromosome, mutationRate)
        pop.append(mutated)
    return pop

def geneticAlgorithm(starting_point, mutationRate, elitismRate, generation_num, population_size):
    coordinates = []
    progress = []
    # citanje podataka iz data_tsp.txt
    with open('data_tsp.txt', 'r') as f:
        for line in f:
            ls = line.split();
            ls[0] = float(ls[0])
            ls[1] = float(ls[1])
            ls[2] = float(ls[2])
            coordinates.append(ls)
    elitism = int(elitismRate * population_size)
    adjacency_matrix = np.zeros(shape=(len(coordinates), len(coordinates)))

    # popunjavanje matrice incidencije
    calculate_distances(adjacency_matrix, coordinates, len(coordinates))

    # PRIKAZ ADJ MATRICE
    # with np.printoptions(threshold=np.inf):
    #     print(len(adjacency_matrix))
    #     print(adjacency_matrix)

    route = initialize_a_route(starting_point, len(coordinates))
    population = create_starting_population(population_size, route)
    progress.append(population_fitness(population, adjacency_matrix)[-1][1])

    for _ in range(generation_num):
        pop_fitness = population_fitness(population, adjacency_matrix)
        selectedPop = selection(pop_fitness, elitism)
        pool = [population[ind] for ind in selectedPop]
        population = breedPopulation(pool, elitism)
        population = mutatePopulation(population, mutationRate, elitism)
        gene_ind,gene_fitness = population_fitness(population, adjacency_matrix)[-1]
        progress.append(gene_fitness)

    # iscrtavanje grafikona
    plt.plot(progress)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()
    return population[gene_ind], gene_fitness


def calculate_distances(adjacency_matrix, coordinates, numOfCities):
    for i in range(0, numOfCities):
        for j in range(0, i):
            adjacency_matrix[i, j] = math.sqrt(
                (coordinates[i][1] - coordinates[j][1]) ** 2 + (coordinates[i][2] - coordinates[j][2]) ** 2)
            adjacency_matrix[j, i] = adjacency_matrix[i, j]


def main():
    best_solution, best_solution_fitness = geneticAlgorithm(starting_point=41, mutationRate= 0.02, elitismRate= 0.15, generation_num= 1000, population_size= 400)
    print("Best route:", best_solution, "\nFitness:", best_solution_fitness)


if __name__ == "__main__":
    main()
