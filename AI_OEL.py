import random
import time


class GeneticAlgorithm:
    def __init__(self, target, population_size, mutation_rate, max_generations):
        # Initialize the main properties of the algorithm
        self.target = target
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.gene_pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz"
        self.population = []
        self.best_solution = None

    def initialize_population(self):
        """Create the starting population of random strings."""
        self.population = []
        for _ in range(self.population_size):
            individual = ''.join(random.choices(self.gene_pool, k=len(self.target)))
            self.population.append(individual)

    def fitness(self, individual):
        """Calculate how close an individual is to the target."""
        # Count matching characters
        #Compares the character c at index i in individual to the character at 
        #the same position in self.target
        match_score = sum(1 for i, c in enumerate(individual) if c == self.target[i])
        # Encourage diversity by penalizing repeated characters
        diversity_penalty = len(set(individual)) / len(self.target)
        return match_score + diversity_penalty

    def select_parents(self):
        """Pick two parents using tournament selection."""
        def tournament():
            competitors = random.sample(self.population, k=5)  # Random 5 individuals
            return max(competitors, key=self.fitness)  # Select the best of 5

        return tournament(), tournament()

    def crossover(self, parent1, parent2):
        """Create a child by mixing genes from two parents."""
        return ''.join(random.choice([gene1, gene2]) for gene1, gene2 in zip(parent1, parent2))

    def mutate(self, chromosome, stagnation):
        """Make random changes to a chromosome."""
        # Double the mutation rate if the algorithm is stuck
        effective_rate = self.mutation_rate * (2 if stagnation else 1)
        return ''.join(gene if random.random() > effective_rate 
            else random.choice(self.gene_pool)
            for gene in chromosome
        )

    def run(self):
        """Run the genetic algorithm."""
        self.initialize_population()
        start_time = time.time()  # Record the start time
        generation_times = []
        best_fitness = 0
        stagnation_counter = 0

        for generation in range(1, self.max_generations + 1):
            gen_start_time = time.time()

            # Evaluate fitness of the population
            self.population.sort(key=self.fitness, reverse=True)
            current_best = self.population[0]
            current_fitness = self.fitness(current_best)

            # Update the best solution found so far
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                self.best_solution = current_best
                stagnation_counter = 0  # Reset stagnation counter
            else:
                stagnation_counter += 1

            print(f"Generation {generation}: Best fitness = {current_fitness}, Best solution = {current_best}")

            # Stop if the target is reached
            if current_fitness >= len(self.target):
                print("Target reached!")
                break

            # Handle stagnation (no improvement for a long time)
            if stagnation_counter >= 50:
                print("Too much stagnation. Reintroducing diversity.")
                self.initialize_population()
                stagnation_counter = 0

            # Create the next generation
            stagnation = stagnation_counter >= 20
            new_population = [self.best_solution]  # Keep the best solution (elitism)
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                child = self.mutate(self.crossover(parent1, parent2), stagnation)
                new_population.append(child)

            self.population = new_population
            generation_times.append(time.time() - gen_start_time)

        # Report performance metrics
        total_time = time.time() - start_time
        avg_gen_time = sum(generation_times) / len(generation_times) if generation_times else 0
        print("\nPerformance Summary:")
        print(f"Total Generations: {generation}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Average Time per Generation: {avg_gen_time:.4f} seconds")
        print(f"Best Solution: {self.best_solution}, Fitness: {best_fitness}")


# Example usage
if __name__ == "__main__":
    target_string = "My name is Ayesha Taufique "
    ga = GeneticAlgorithm(target=target_string, population_size=500, mutation_rate=0.05, max_generations=60)
    ga.run()
