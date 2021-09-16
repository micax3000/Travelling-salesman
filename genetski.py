import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator
import random
#--------------------------------------------------------
def initialize_a_route(starting_point,length):
    route = np.array(range(1,length+1), dtype = int)
    route = np.delete(route, starting_point-1,0,)
    route = np.concatenate(([starting_point],route,[starting_point]))
    return route
#izracunava distancu izmedju svih gradova
def fitness(member,adjacency_matrix):
    fitness = 0
    member_len = len(member)
    for i in range(0,member_len-1):
        x = member[i]
        y = member[i+1]
        fitness += adjacency_matrix[x-1][y-1]

    return fitness
#izracunava duzinu svih puteva u odredjenoj generaciji i sortira ih u opadajucem redosledu
#put sa najmanjom duzinom je na kraju liste
def population_fitness(population,adjacency_matrix):
    pop_fitness = {}
    
    for i in range(0,len(population)):
        pop_fitness[i] = fitness(population[i],adjacency_matrix)
    sorted_list = sorted(pop_fitness.items(), key = operator.itemgetter(1), reverse = True)
    return sorted_list

def make_a_list_copy(list):
    a = list[0]
    list = np.delete(list,-1)
    list = np.delete(list,0)
    b = np.copy(list)
    random.shuffle(b)
    b = np.insert(b,0,a)
    b = np.append(b,a)
    return b

def create_starting_population(size,route):
    population = []

    for i in range(size):
        b = make_a_list_copy(route)
        population.append(b)
    return population
#bira jedinke za ukrstanje i vraca njihove indekse
#indeksi jedinki se mogu pojaviti vise puta
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['probability'] = 100 * df.cum_sum / df.Fitness.sum()
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[len(popRanked)-1-i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for j in range(i, len(popRanked)):
            if pick <= df.iat[j, 3]:
                selectionResults.append(popRanked[j][0])
                break
    return selectionResults
#ukrstavanje dve jedinke
#dete uzima niz gena ordredjene duzine od prvog roditelja
#sve ostale gene do len(a)(ili len(b)) uzima od drugog roditelja sa tim da se geni ne smeju ponavljati
def breed(a,b):
    child = []
    part1 = []
    part2 = []
    geneA = 0
    geneB = 0
    while geneA == 0 or geneB == 0:
        geneA = int(random.random() * (len(a) - 1))
        geneB = int(random.random() * (len(a) - 1))

    startGene = min(geneA,geneB)
    endGene = max(geneA,geneB)
    part1.append(a[0])
    for i in range(startGene,endGene):
        part1.append(a[i])

    part2 = [item for item in b if item not in part1]
    part2.append(a[0])

    child = part1 + part2
    return child
#vrsi ukrstavanje populacije sa elitizmom
#ukrstanje po parovima
def breedPopulation(pool,elitism):
    bredPopulation = []

    for i in range(elitism):
        bredPopulation.append(pool[i])

    for i in range(0,len(pool)-elitism):
        child = breed(pool[i],pool[len(pool)-i-1]);
        bredPopulation.append(child)
    return bredPopulation
#mutiranje jedinki predstavlja zamenu mesta nasumicno izabranih gena
def mutate(chromosome,mutationRate):
    for indx in range(1,len(chromosome)-1):
        if(random.random() < mutationRate):
            indx2 = random.randint(1, len(chromosome)-2)

            temp = chromosome[indx]
            chromosome[indx] = chromosome[indx2]
            chromosome[indx2] = temp

    return chromosome
#mutira celu populaciju
def mutatePopulation(matingPool,mutationRate,elitism):
    pop = []
    for i in range(elitism):
        pop.append(matingPool[i])
    for i in range(elitism,len(matingPool)):
        chromosome = matingPool[i]
        mutated = mutate(chromosome,mutationRate)
        pop.append(mutated)
    return pop
#uzima niz indeksa jedinki izabranih u select funkciji i vraca te jedinke
def matingPool(population, selectionPool):
    matingPool = []
    for i in range(len(selectionPool)):
        indx = selectionPool[i]
        matingPool.append(population[indx])

    return matingPool

def geneticAlgorithm(starting_point,mutationRate,elitismRate,generation_num,population_size):
    coordinates = []
    #citanje podataka iz data_tsp.txt
    with open('data_tsp.txt', 'r') as f:
        for line in f:
            ls = line.split();
            ls[0] = float(ls[0])
            ls[1] = float(ls[1])
            ls[2] = float(ls[2])
            coordinates.append(ls)
    elitism = int(elitismRate * population_size)
    numOfCities = len(coordinates)
    adjacency_matrix = np.zeros(shape=(numOfCities, numOfCities))


    #popunjavanje matrice incidencije
    for i in range(0, numOfCities):
        for j in range(0, i):
            adjacency_matrix[i, j] = math.sqrt( (coordinates[i][1] - coordinates[j][1]) ** 2 + (coordinates[i][2] - coordinates[j][2]) ** 2)
            adjacency_matrix[j, i] = adjacency_matrix[i, j]

    # PRIKAZ ADJ MATRICE
    # with np.printoptions(threshold=np.inf):
    #     print(len(adjacency_matrix))
    #     print(adjacency_matrix)

    route = initialize_a_route(starting_point, numOfCities)
    population = create_starting_population(population_size,route)
    progress = []
    progress.append(population_fitness(population,adjacency_matrix)[-1][1])

    for i in range(0,generation_num):
        pop_fitness = population_fitness(population, adjacency_matrix)
        selectedPop = selection(pop_fitness,elitism)
        pool = matingPool(population,selectedPop)
        population = breedPopulation(pool,elitism)
        population = mutatePopulation(population,mutationRate,elitism)
        best_solution = population_fitness(population, adjacency_matrix)[-1]
        progress.append(best_solution[1])

    #iscrtavanje grafikona
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    return population[best_solution[0]],best_solution[1]
def main():
    (best_solution,best_solution_fitness) = geneticAlgorithm(41,0.02,0.15,1000,200)
    print("Best route:",best_solution,"\nFitness:",best_solution_fitness)

if __name__== "__main__":
  main()