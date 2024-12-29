#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <unistd.h> // For getcwd on Unix-like systems

#define SUBPOP_SIZE 500
#define NUM_SUBPOPULATIONS 15
#define MAX_GENERATIONS 100// Define the maximum number of generations
#define MUTATION_PROBABILITY 0.45 // Set this to your desired mutation probability
#define MIGRATION_INTERVAL 10  // Every 10 generations
#define MIGRANTS_PER_POP 150 // Migrate top 5 individuals

void freeHostAndDeviceMemory(float *h_fitnessP, float *h_fitnessW, float *h_fitness, int *h_populations, int *flatMatrix1, int *flatMatrix2, int *d_populationsP, int *d_populationsW, int *d_flatMatrix1, int *d_flatMatrix2, float *d_fitnessP, float *d_fitnessW, curandState *d_state) {
    // Free host memory
    if (h_fitnessP != NULL) free(h_fitnessP);
    if (h_fitnessW != NULL) free(h_fitnessW);
    if (h_fitness != NULL) free(h_fitness);
    if (h_populations != NULL) free(h_populations);
    if (flatMatrix1 != NULL) free(flatMatrix1);
    if (flatMatrix2 != NULL) free(flatMatrix2);
    

    // Free device memory
    cudaError_t err;
    if (d_populationsP != NULL) {
        err = cudaFree(d_populationsP);
        if (err != cudaSuccess) printf("Error freeing d_populationsP: %s\n", cudaGetErrorString(err));
    }
    if (d_populationsW != NULL) {
        err = cudaFree(d_populationsW);
        if (err != cudaSuccess) printf("Error freeing d_populationsW: %s\n", cudaGetErrorString(err));
    }
    if (d_flatMatrix1 != NULL) {
        err = cudaFree(d_flatMatrix1);
        if (err != cudaSuccess) printf("Error freeing d_flatMatrix1: %s\n", cudaGetErrorString(err));
    }
    if (d_flatMatrix2 != NULL) {
        err = cudaFree(d_flatMatrix2);
        if (err != cudaSuccess) printf("Error freeing d_flatMatrix2: %s\n", cudaGetErrorString(err));
    }
    if (d_fitnessP != NULL) {
        err = cudaFree(d_fitnessP);
        if (err != cudaSuccess) printf("Error freeing d_fitnessP: %s\n", cudaGetErrorString(err));
    }
    if (d_fitnessW != NULL) {
        err = cudaFree(d_fitnessW);
        if (err != cudaSuccess) printf("Error freeing d_fitnessW: %s\n", cudaGetErrorString(err));
    }
    if (d_state != NULL) {
        err = cudaFree(d_state);
        if (err != cudaSuccess) printf("Error freeing d_state: %s\n", cudaGetErrorString(err));
    }
}

void extractMatrices(FILE *file, int ***matrix1, int ***matrix2, int *size) {
    // Read the size of the matrices
    fscanf(file, "%d", size);

    // Allocate memory for the matrices
    *matrix1 = (int **)malloc(*size * sizeof(int *));
    *matrix2 = (int **)malloc(*size * sizeof(int *));

    for (int i = 0; i < *size; i++) {
        (*matrix1)[i] = (int *)malloc(*size * sizeof(int));
        (*matrix2)[i] = (int *)malloc(*size * sizeof(int));
    }

    // Read the matrices
    for (int i = 0; i < *size; i++) {
        for (int j = 0; j < *size; j++) {
            fscanf(file, "%d", &(*matrix1)[i][j]);
        }
    }

    for (int i = 0; i < *size; i++) {
        for (int j = 0; j < *size; j++) {
            fscanf(file, "%d", &(*matrix2)[i][j]);
        }
    }
}

int *flattenMatrix(int **matrix, int size) {
    int *flatMatrix = (int *)malloc(size * size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            flatMatrix[i * size + j] = matrix[i][j];
        }
    }
    return flatMatrix;
}

__global__ void init_random(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void initialize_population_kernel(int *populations, curandState *state, int subpop_size, int num_locations, int num_subpopulations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= subpop_size * num_subpopulations) return;

    int startIdx = idx * num_locations;
    curandState localState = state[idx];

    for (int i = 0; i < num_locations; ++i) {
        populations[startIdx + i] = i;
    }

    for (int i = num_locations - 1; i > 0; --i) {
        int j = curand(&localState) % (i + 1);
        int temp = populations[startIdx + i];
        populations[startIdx + i] = populations[startIdx + j];
        populations[startIdx + j] = temp;
    }

    state[idx] = localState;
}

__device__ int getNonDuplicate(int gene, const int *segment, int point1, int point2) {
    for (int i = point1; i <= point2; ++i) {
        if (gene == segment[i]) {
            gene = segment[point1 + (i - point1)]; // Map to corresponding position
            i = point1 - 1; // Restart checking
        }
    }
    return gene;
}

__global__ void pmx_crossover_kernel(int *populations, curandState *state, int subpop_size, int num_locations, int num_subpopulations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= subpop_size * num_subpopulations) return;

    curandState localState = state[idx];
    int partnerIdx = (idx + subpop_size / 2) % subpop_size + (idx / subpop_size) * subpop_size;

    int point1 = curand(&localState) % (num_locations - 1);
    int point2 = curand(&localState) % num_locations;
    while (point1 == point2) {
        point2 = curand(&localState) % num_locations;
    }
    if (point1 > point2) {
        int temp = point1;
        point1 = point2;
        point2 = temp;
    }

    int startIdx1 = idx * num_locations;
    int startIdx2 = partnerIdx * num_locations;
    int* offspring = (int*)malloc(num_locations * sizeof(int));
    if (offspring == NULL) return; // Check for successful allocation
    

    // Flags to mark which genes have been taken
    bool* taken = (bool*)malloc(num_locations * sizeof(bool));
    if (taken == NULL) {
        free(offspring); // Free previously allocated memory
        return; // Check for successful allocation
    }
    for (int i = 0; i < num_locations; ++i) {
        taken[i] = false;
    }

    // Step 1: Copy the crossover segment from the first parent to offspring
    for (int i = point1; i <= point2; ++i) {
        offspring[i] = populations[startIdx1 + i];
        taken[offspring[i]] = true;
    }

    // Step 2: Fill the remaining positions with genes from the second parent, skipping over taken genes
    for (int i = 0, j = 0; i < num_locations; ++i) {
        if (!(i >= point1 && i <= point2)) {
            while (taken[populations[startIdx2 + j]]) {
                j++;
            }
            offspring[i] = populations[startIdx2 + j];
            taken[offspring[i]] = true;
        }
    }

    // Copy offspring back to the population and free dynamic memory
    for (int i = 0; i < num_locations; ++i) {
        populations[startIdx1 + i] = offspring[i];
    }
    free(offspring);
    free(taken);

    state[idx] = localState;
}

__global__ void pmx_crossover_and_mutation_kernel(int *populationsP, int *populationsW, curandState *state, int subpop_size, int num_locations, int num_subpopulations, float mutation_probability) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= subpop_size * num_subpopulations) return;

    curandState localState = state[idx];
    int partnerIdx = (idx + subpop_size / 2) % subpop_size + (idx / subpop_size) * subpop_size;

    // Crossover points
    int point1 = curand(&localState) % (num_locations - 1);
    int point2 = curand(&localState) % num_locations;
    while (point1 == point2) {
        point2 = curand(&localState) % num_locations;
    }
    if (point1 > point2) {
        int temp = point1;
        point1 = point2;
        point2 = temp;
    }

    int startIdx1 = idx * num_locations;
    int startIdx2 = partnerIdx * num_locations;

    int* offspring = (int*)malloc(num_locations * sizeof(int));
    if (!offspring) return; // Check for successful allocation

    bool* taken = (bool*)malloc(num_locations * sizeof(bool));
    if (!taken) {
        free(offspring); // Free the already allocated memory
        return; // Check for successful allocation
    }

    // Initialize taken array
    for (int i = 0; i < num_locations; ++i) {
        taken[i] = false;
    }

    // Copy the crossover segment from the first parent to offspring
    for (int i = point1; i <= point2; ++i) {
        offspring[i] = populationsP[startIdx1 + i];
        taken[offspring[i]] = true;
    }

    // Fill the remaining positions with genes from the second parent, avoiding duplicates
    for (int i = 0, j = 0; i < num_locations; ++i) {
        if (i >= point1 && i <= point2) continue;
        while (taken[populationsP[startIdx2 + j]]) {
            j++;
        }
        offspring[i] = populationsP[startIdx2 + j];
        taken[offspring[i]] = true;
    }

    // Mutation step
    if (curand_uniform(&localState) < mutation_probability) {
        int gene1 = curand(&localState) % num_locations;
        int gene2 = curand(&localState) % num_locations;
        while (gene1 == gene2) {
            gene2 = curand(&localState) % num_locations;
        }
        // Swap two genes for mutation
        int temp = offspring[gene1];
        offspring[gene1] = offspring[gene2];
        offspring[gene2] = temp;
    }

    // Copy offspring to the working pool W
    for (int i = 0; i < num_locations; ++i) {
        populationsW[startIdx1 + i] = offspring[i];
    }

    // Free dynamically allocated memory
    free(offspring);
    free(taken);

    // Update the state
    state[idx] = localState;
}

__global__ void mutation_kernel(int *populations, curandState *state, float mutation_probability, int subpop_size, int num_locations, int num_subpopulations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= subpop_size * num_subpopulations) return;

    curandState localState = state[idx];
    float rnd = curand_uniform(&localState);

    if (rnd < mutation_probability) {
        int startIdx = idx * num_locations;

        // Randomly select two different positions for mutation
        int pos1 = curand(&localState) % num_locations;
        int pos2 = curand(&localState) % num_locations;
        while (pos1 == pos2) { // Ensure the two positions are different
            pos2 = curand(&localState) % num_locations;
        }

        // Swap the genes at pos1 and pos2
        int temp = populations[startIdx + pos1];
        populations[startIdx + pos1] = populations[startIdx + pos2];
        populations[startIdx + pos2] = temp;
    }

    state[idx] = localState; // Update the state
}

__global__ void calculate_fitness(int *populations, float *fitness, int *distance_matrix, int *flow_matrix, int subpop_size, int num_locations, int num_subpopulations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= subpop_size * num_subpopulations) return;

    int startIdx = idx * num_locations;
    float individual_fitness = 0.0f;

    for (int i = 0; i < num_locations; ++i) {
        for (int j = 0; j < num_locations; ++j) {
            int facility1 = populations[startIdx + i];
            int facility2 = populations[startIdx + j];
            int distance = distance_matrix[i * num_locations + j]; // Assuming distance_matrix is linearized
            int flow = flow_matrix[facility1 * num_locations + facility2]; // Assuming flow_matrix is linearized

            individual_fitness += distance * flow;
        }
    }

    fitness[idx] = individual_fitness;
}

float calculateFitnessManual(int *solution, int *distance_matrix, int *flow_matrix, int size) {
    float fitness = 0.0f;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int facility1 = solution[i];
            int facility2 = solution[j];
            int distance = distance_matrix[i * size + j];
            int flow = flow_matrix[facility1 * size + facility2];
            fitness += distance * flow;
        }
    }
    return fitness;
}

void printSubpopulations(int *populations, int num_subpopulations, int subpop_size, int num_locations, int toPrint) {
    printf("Printing %d subpopulation(s):\n", toPrint);
    for (int i = 0; i < toPrint; ++i) {
        printf("Subpopulation %d, First Individual: ", i + 1);
        for (int j = 0; j < num_locations; ++j) {
            printf("%d ", populations[i * subpop_size * num_locations + j] + 1); // +1 for 1-indexed printing
        }
        printf("\n");
    }
}

__device__ int select_partner(int idx, curandState *state, int subpop_size) {
    int partner = idx;
    while (partner == idx) {
        partner = curand_uniform(&state[idx]) * subpop_size;
    }
    return partner;
}

__global__ void replace_with_better_kernel(int *populationsP, int *populationsW, float *fitnessP, float *fitnessW, int subpop_size, int num_locations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= subpop_size) return;

    // If the new individual has better fitness, replace the old one
    if (fitnessW[idx] < fitnessP[idx]) {
        int startIdx = idx * num_locations;
        for (int i = 0; i < num_locations; ++i) {
            populationsP[startIdx + i] = populationsW[startIdx + i];
        }
        fitnessP[idx] = fitnessW[idx];
    }
}

void printSubpopulation(const int* subpopulation, int subpop_size, int num_locations) {
    for (int i = 0; i < subpop_size; ++i) {
        printf("Individual %d: ", i + 1);
        fflush(stdout);
        for (int j = 0; j < num_locations; ++j) {
            printf("%d ", subpopulation[i * num_locations + j] + 1); // +1 for 1-indexed printing
            fflush(stdout);
        }
        printf("\n");
    }
}

__global__ void migrate_individuals(int *populations, int subpop_size, int num_locations, int num_subpopulations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_population = subpop_size * num_subpopulations;

    if (idx >= total_population) return;

    // Calculate source and target subpopulation indices
    int source_subpop = idx / subpop_size;
    int target_subpop = (source_subpop + 1) % num_subpopulations;  // Ring topology

    // Only the first MIGRANTS_PER_POP individuals in each subpopulation participate in migration
    if (idx % subpop_size < MIGRANTS_PER_POP) {
        int source_idx = idx;
        int target_idx = target_subpop * subpop_size + (idx % subpop_size);

        // Swap individuals
        for (int i = 0; i < num_locations; ++i) {
            int temp = populations[source_idx * num_locations + i];
            populations[source_idx * num_locations + i] = populations[target_idx * num_locations + i];
            populations[target_idx * num_locations + i] = temp;
        }
    }
}

__global__ void selective_migrate_individuals(int *populations, float *fitness, int subpop_size, int num_locations, int num_subpopulations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int subpop_idx = idx / subpop_size;
    int within_subpop_idx = idx % subpop_size;

    if (subpop_idx >= num_subpopulations || within_subpop_idx >= MIGRANTS_PER_POP) return;

    // Calculate target subpopulation in a ring topology
    int target_subpop = (subpop_idx + 1) % num_subpopulations;

    // Indices for source and target
    int source_idx = subpop_idx * subpop_size + within_subpop_idx;
    int target_idx = target_subpop * subpop_size + within_subpop_idx;

    // Swap individuals between source and target
    for (int i = 0; i < num_locations; ++i) {
        int temp = populations[source_idx * num_locations + i];
        populations[source_idx * num_locations + i] = populations[target_idx * num_locations + i];
        populations[target_idx * num_locations + i] = temp;
    }
}

__global__ void simple_migration(int *populations, int num_locations, int num_subpopulations, int subpop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_locations) return; // Only migrate one individual per subpopulation

    for (int i = 0; i < num_subpopulations - 1; i++) {
        int source_idx = i * subpop_size * num_locations + idx;
        int target_idx = (i + 1) * subpop_size * num_locations + idx;

        // Simple swap between first individuals of consecutive subpopulations
        int temp = populations[source_idx];
        populations[source_idx] = populations[target_idx];
        populations[target_idx] = temp;
    }
}

int main() {
    
    const char *filenames[] = {"bur26a.dat", "bur26b.dat", "bur26c.dat", "bur26d.dat",
    "bur26e.dat", "bur26f.dat", "bur26g.dat", "bur26h.dat",
    "chr12a.dat", "chr12b.dat", "chr12c.dat", "chr15a.dat",
        "chr15b.dat", "chr15c.dat", "chr18a.dat", "chr18b.dat",
        "chr20a.dat", "chr20b.dat", "chr20c.dat", "chr22a.dat",
        "chr22b.dat", "chr25a.dat", "els19.dat", "esc16a.dat",
        "esc16b.dat", "esc16c.dat", "esc16d.dat", "esc16e.dat",
        "esc16f.dat", "esc16g.dat", "esc16h.dat", "esc16i.dat",
        "esc16j.dat", "esc32a.dat", "esc32b.dat", "esc32c.dat",
        "esc32d.dat", "esc32e.dat", "esc32g.dat", "esc32h.dat",
        "esc64a.dat", "esc128.dat", "had12.dat", "had14.dat",
        "had16.dat", "had18.dat", "had20.dat", "kra30a.dat",
        "kra30b.dat", "kra32.dat", "lipa20a.dat", "lipa20b.dat",
        "lipa30a.dat", "lipa30b.dat", "lipa40a.dat", "lipa40b.dat",
        "lipa50a.dat", "lipa50b.dat", "lipa60a.dat", "lipa60b.dat",
        "lipa70a.dat", "lipa70b.dat", "lipa80a.dat", "lipa80b.dat",
        "lipa90a.dat", "lipa90b.dat", "nug12.dat", "nug14.dat",
        "nug15.dat", "nug16a.dat", "nug16b.dat", "nug17.dat",
        "nug18.dat", "nug20.dat", "nug21.dat", "nug22.dat",
        "nug24.dat", "nug25.dat", "nug27.dat", "nug28.dat",
        "nug30.dat", "rou12.dat", "rou15.dat", "rou20.dat",
        "scr12.dat", "scr15.dat", "scr20.dat", "sko42.dat",
        "sko49.dat", "sko56.dat", "sko64.dat", "sko72.dat",
        "sko81.dat", "sko90.dat", "sko100a.dat", "sko100b.dat",
        "sko100c.dat", "sko100d.dat", "sko100e.dat", "sko100f.dat",
        "ste36a.dat", "ste36b.dat", "ste36c.dat", "tai12a.dat",
        "tai12b.dat", "tai15a.dat", "tai15b.dat", "tai17a.dat",
        "tai20a.dat", "tai20b.dat", "tai25a.dat", "tai25b.dat",
        "tai30a.dat", "tai30b.dat", "tai35a.dat", "tai35b.dat",
        "tai40a.dat", "tai40b.dat", "tai50a.dat", "tai50b.dat",
        "tai60a.dat", "tai60b.dat", "Tai64c.dat", "tai80a.dat",
        "tai80b.dat", "tai100a.dat", "tai100b.dat", "tai150b.dat",
        "tai256c.dat", "tho30.dat", "tho40.dat", "tho150.dat",
        "wil50.dat", "wil100.dat"
};
    const double optimal[] = {5426670, 3817852, 5426795, 3821225, 
    5386879, 3782044, 10117172, 7098658,
    9552, 9742, 11156, 9896,
    7990, 9504, 11098, 1534,
    2192, 2298, 14142, 6156,
    6194, 3796, 17212548, 68,
    292, 160, 16, 28,
    0, 26, 996, 14,
    8, 130, 168, 642,
    200, 2, 6, 438,
    116, 64, 1652, 2724,
    3720, 5358, 6922, 88900,
    91420, 88700, 3683, 27076,
    13178, 151426, 31538, 476581,
    62093, 1210244, 107218, 2520135,
    169755, 4603200, 253195, 7763962,
    360630, 12490441, 578, 1014,
    1150, 1610, 1240, 1732,
    1930, 2570, 2438, 3596,
    3488, 3744, 5234, 5166,
    6124, 235528, 354210, 725522,
    31410, 51140, 110030, 15812,
    23386, 34458, 48498, 66256,
    90998, 115534, 152002, 153890,
    147862, 149576, 149150, 149036,
    9526, 15852, 8239110, 224416,
    39464925, 388214, 51765268, 491812,
    703482, 122455319, 1167256, 344355646,
    1818146, 637117113, 2422002, 283315445,
    3139370, 637250948, 4938796, 458821517,
    7205962, 608215054, 1855928, 13499184,
    818415043, 21052466, 1185996137, 498896643,
    44759294, 149936, 240516, 8133398,
    48816, 273038};
    
    FILE *resultsFile = fopen("results.txt", "w");
    if (resultsFile == NULL) {
        printf("Error opening results file.\n");
        fflush(stdout);
        return 1;
    }

    int num_files = sizeof(filenames) / sizeof(filenames[0]);

   for (int file_idx = 0; file_idx < num_files; file_idx++) {
        FILE *file = fopen(filenames[file_idx], "r");
        if (!file) {
            fprintf(stderr, "Error opening file %s\n", filenames[file_idx]);
            fflush(stdout);
            continue; // Skip to the next file
        }


        int **matrix1, **matrix2, size;
        extractMatrices(file, &matrix1, &matrix2, &size);
        fclose(file);

        printf("Instance: %s\n", filenames[file_idx]);
        fflush(stdout);

        int num_locations = size;


        // Flatten the 2D matrices to 1D arrays
        int *flatMatrix1 = flattenMatrix(matrix1, size);
        int *flatMatrix2 = flattenMatrix(matrix2, size);

        int *d_populationsP, *d_populationsW, *d_flatMatrix1, *d_flatMatrix2;
        size_t pop_size_bytes = NUM_SUBPOPULATIONS * SUBPOP_SIZE * num_locations * sizeof(int);
        size_t matrix_size_bytes = size * size * sizeof(int);
        size_t fitness_size_bytes = NUM_SUBPOPULATIONS * SUBPOP_SIZE * sizeof(float);
        float *d_fitnessP, *d_fitnessW; // Need to allocate memory for these
        float *h_fitnessP = (float *)malloc(fitness_size_bytes); // Host array for fitness of Population P
        float *h_fitnessW = (float *)malloc(fitness_size_bytes); // Host array for fitness of Population W

        
        cudaMalloc(&d_populationsP, pop_size_bytes);
        cudaMalloc(&d_populationsW, pop_size_bytes);
        cudaMalloc(&d_flatMatrix1, matrix_size_bytes);
        cudaMalloc(&d_flatMatrix2, matrix_size_bytes);
        cudaMalloc(&d_fitnessP, fitness_size_bytes);
        cudaMalloc(&d_fitnessW, fitness_size_bytes);

        // Copy flattened matrices to device
        cudaMemcpy(d_flatMatrix1, flatMatrix1, matrix_size_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_flatMatrix2, flatMatrix2, matrix_size_bytes, cudaMemcpyHostToDevice);
        
        float *h_fitness = (float *)malloc(NUM_SUBPOPULATIONS * SUBPOP_SIZE * sizeof(float));
        int *h_populations = (int *)malloc(NUM_SUBPOPULATIONS * SUBPOP_SIZE * num_locations * sizeof(int));
        float best_fitness = FLT_MAX; // Track the best fitness; start with the highest possible value
        int best_solution[num_locations]; // To store the best solution

        curandState *d_state;
        cudaMalloc(&d_state, NUM_SUBPOPULATIONS * SUBPOP_SIZE * sizeof(curandState));

        float maxTimeInSeconds = 10.0f; // 10 seconds as an example
        float maxTimeInMilliseconds = maxTimeInSeconds * 1000.0f; // Convert seconds to milliseconds



        // Initialize random states for each thread
        init_random<<<NUM_SUBPOPULATIONS, SUBPOP_SIZE>>>(d_state, time(NULL));
        cudaDeviceSynchronize();

        // Main loop for generations
        int generation = 0;
        float elapsedTime = 0;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        initialize_population_kernel<<<NUM_SUBPOPULATIONS, SUBPOP_SIZE>>>(d_populationsP, d_state, SUBPOP_SIZE, num_locations, NUM_SUBPOPULATIONS);
        cudaDeviceSynchronize();
        //120000
        
        while (elapsedTime < 120000) {
            //int* h_subpopulationsW_before = (int*)malloc(NUM_SUBPOPULATIONS * SUBPOP_SIZE * num_locations * sizeof(int));
            //int* h_subpopulationsW_after = (int*)malloc(NUM_SUBPOPULATIONS * SUBPOP_SIZE * num_locations * sizeof(int));

            //cudaMemcpy(h_subpopulationsW_before, d_populationsW, NUM_SUBPOPULATIONS * SUBPOP_SIZE * num_locations * sizeof(int), cudaMemcpyDeviceToHost);

            
            // Print PopulationW before crossover
            /*
            printf("PopulationW before crossover:\n");
            fflush(stdout);
            for (int i = 0; i < NUM_SUBPOPULATIONS; ++i) {
                printf("Subpopulation %d:\n", i + 1);
                fflush(stdout);
                printSubpopulation(&h_subpopulationsW_before[i * SUBPOP_SIZE * num_locations], SUBPOP_SIZE, num_locations);
            }
            */
            // Perform crossover and mutation from populationsP to populationsW
            pmx_crossover_and_mutation_kernel<<<NUM_SUBPOPULATIONS, SUBPOP_SIZE>>>(
                d_populationsP, d_populationsW, d_state, SUBPOP_SIZE, num_locations, NUM_SUBPOPULATIONS, MUTATION_PROBABILITY
            );
            cudaDeviceSynchronize();

            if (generation % 10 == 0) {  // Every 10 generations
                simple_migration<<<1, num_locations>>>(d_populationsP, num_locations, NUM_SUBPOPULATIONS, SUBPOP_SIZE);
                cudaDeviceSynchronize();
            }



            //cudaMemcpy(h_subpopulationsW_after, d_populationsW, NUM_SUBPOPULATIONS * SUBPOP_SIZE * num_locations * sizeof(int), cudaMemcpyDeviceToHost);
            /*
            printf("PopulationW after crossover:\n");
            fflush(stdout);
            for (int i = 0; i < NUM_SUBPOPULATIONS; ++i) {
                printf("Subpopulation %d:\n", i + 1);
                fflush(stdout);
                printSubpopulation(&h_subpopulationsW_after[i * SUBPOP_SIZE * num_locations], SUBPOP_SIZE, num_locations);
            }
            free(h_subpopulationsW_before);
            free(h_subpopulationsW_after);
            */
            // Calculate fitness for populationsW
            calculate_fitness<<<NUM_SUBPOPULATIONS, SUBPOP_SIZE>>>(
                d_populationsW, d_fitnessW, d_flatMatrix1, d_flatMatrix2, SUBPOP_SIZE, num_locations, NUM_SUBPOPULATIONS
            );
            cudaDeviceSynchronize();

            calculate_fitness<<<NUM_SUBPOPULATIONS, SUBPOP_SIZE>>>(
                        d_populationsP, d_fitnessP, d_flatMatrix1, d_flatMatrix2, SUBPOP_SIZE, num_locations, NUM_SUBPOPULATIONS
                    );
            cudaDeviceSynchronize();

            // Replace individuals in populationsP with better counterparts from populationsW
            replace_with_better_kernel<<<NUM_SUBPOPULATIONS, SUBPOP_SIZE>>>(
                d_populationsP, d_populationsW, d_fitnessP, d_fitnessW, SUBPOP_SIZE, num_locations
            );
            cudaDeviceSynchronize();

            // Copy fitness values back to host to find the best solution
            cudaMemcpy(h_fitnessP, d_fitnessP, fitness_size_bytes, cudaMemcpyDeviceToHost);

            // Find the best solution and its fitness
            float current_fitness;
            int *current_solution = (int *)malloc(num_locations * sizeof(int));
            for (int i = 0; i < NUM_SUBPOPULATIONS * SUBPOP_SIZE; ++i) {
                cudaMemcpy(&current_fitness, d_fitnessP + i, sizeof(float), cudaMemcpyDeviceToHost);
                if (current_fitness < best_fitness) {
                    cudaMemcpy(current_solution, d_populationsP + i * num_locations, num_locations * sizeof(int), cudaMemcpyDeviceToHost);
                    best_fitness = current_fitness;
                    memcpy(best_solution, current_solution, num_locations * sizeof(int));  // Ensure you've defined best_solution at host
                }
            }
            
            free(current_solution);
            generation++;

            // Update elapsed time
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
        }
        //printf("%d\n",generation);
        //fflush(stdout);
        double gap = (best_fitness - optimal[file_idx]) / optimal[file_idx] * 100;
        printf("Best Fitness: %0.0f, Optimal: %0.0f, Time: 120 sec, GAP: %0.3f%%\n", best_fitness, optimal[file_idx], gap);
        fflush(stdout);
        fprintf(resultsFile, "Instance: %s, Best Fitness: %0.0f, Gap: %0.3f%%, Time: 120 sec\n", filenames[file_idx], best_fitness, gap);
        printf("\n");
        fflush(stdout);
        printf("Best Solution: ");
        fflush(stdout);
        for (int i = 0; i < num_locations; ++i) {
            printf("%d ", best_solution[i] + 1);  // +1 if you need 1-indexed output
            fflush(stdout);
        }
        printf("\nBest Fitness: %.2f\n", best_fitness);
        fflush(stdout);

        // Print the best solution and its fitness after all generations
        /*printf("Best Solution across all generations: ");
        fflush(stdout);
        for (int i = 0; i < num_locations; ++i) {
            printf("%d ", best_solution[i]+1);
            fflush(stdout);
        }
        */
        //float xd= calculateFitnessManual(best_solution, flatMatrix1, flatMatrix2, size);
        //printf("\nBest Fitness: %.2f\n", xd);
        //printf("");
        //fflush(stdout);

        //int solution[] = {2,9,10,1,11,4,5,6,7,0,3,8}; // Your given solution array
        //float fitness = calculateFitnessManual(solution, flatMatrix1, flatMatrix2, size);
        //printf("Manual Fitness Calculation xd: %.2f\n", fitness);
        //fflush(stdout);

        freeHostAndDeviceMemory(h_fitnessP, h_fitnessW, h_fitness, h_populations, flatMatrix1, flatMatrix2, d_populationsP, d_populationsW, d_flatMatrix1, d_flatMatrix2, d_fitnessP, d_fitnessW, d_state);


        cudaEventDestroy(start);
        cudaEventDestroy(stop);

    
    }
    fclose(resultsFile);
    return 0;
}


