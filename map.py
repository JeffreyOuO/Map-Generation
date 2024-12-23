import random
from PIL import Image
from collections import deque
import math


def initialize_map(size):
    return [[random.choice([0,1]) for _ in range(size)] for _ in range(size)]



def fitness(map):
    size = len(map)
    fitness_score = 0
    visited = [[False] * size for _ in range(size)]

    def bfs(x, y, value):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque([(x, y)])
        visited[x][y] = True
        count = 1

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size and not visited[nx][ny]:
                    if map[nx][ny] == value:
                        visited[nx][ny] = True
                        queue.append((nx, ny))
                        count += 1
        return count

    for i in range(size):
        for j in range(size):
            if not visited[i][j] and map[i][j]!=2:
                value = map[i][j]
                cluster_size = bfs(i, j, value)
                fitness_score += math.exp(cluster_size)

    return fitness_score

def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, fitness in enumerate(fitnesses):
        current += fitness
        if current > pick:
            return population[i]
    return population[-1]

def crossover(map1, map2):
    crossover_point = random.randint(0, len(map1) - 1)
    new_map = [row[:crossover_point] + map2[i][crossover_point:] for i, row in enumerate(map1)]
    return new_map

def mutate(map, mutation_rate):
    for i in range(len(map)):
        for j in range(len(map[i])):
            if random.random() < mutation_rate:
                map[i][j] = random.choice([0, 1])
    return map

def genetic_algorithm(generations, population_size, map_size, mutation_rate):
    population = [initialize_map(map_size) for _ in range(population_size)]
    
    for generation in range(generations):
        fitnesses = [fitness(map) for map in population]
        parents = [select(population, fitnesses) for _ in range(population_size // 2)]
        
        new_population = []
        for i in range(0, len(parents)-1, 2):
            child1 = crossover(parents[i], parents[i + 1])
            child2 = crossover(parents[i + 1], parents[i])
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        
        population = new_population

    fitnesses = [fitness(map) for map in population]
    best_index = fitnesses.index(max(fitnesses))
    print(fitnesses[best_index])
    return population[best_index]

map_size = 10
population_size = 10
generations = 100000
mutation_rate = 0.2

best_map = genetic_algorithm(generations, population_size, map_size, mutation_rate)

def replace_some_ones_with_three(map, replace_fraction=0.1):
    ones_positions = [(i, j) for i in range(len(map)) for j in range(len(map[i])) if map[i][j] == 1]
    num_to_replace = int(len(ones_positions) * replace_fraction)
    positions_to_replace = random.sample(ones_positions, num_to_replace)
    for i, j in positions_to_replace:
        map[i][j] = 2
    return map

best_map = replace_some_ones_with_three(best_map, replace_fraction=0.1)

for row in best_map:
    print(" ".join(str(cell) for cell in row))

image_map = {
    2: 'riverstone.png',
    1: 'river.png',
    0: 'grass.png',
}

tile_size = 32

def generate_image(map_data, image_map, tile_size):
    rows = len(map_data)
    cols = len(map_data[0])

    img = Image.new('RGB', (cols * tile_size, rows * tile_size))

    for i in range(rows):
        for j in range(cols):
            tile_value = map_data[i][j]
            tile_image = Image.open(image_map[tile_value])
            tile_image = tile_image.resize((tile_size, tile_size))
            img.paste(tile_image, (j * tile_size, i * tile_size))

    return img

img = generate_image(best_map, image_map, tile_size)
img.show()
img.save('generated_map.png')
