#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
#include <float.h>

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

void shuffle(int array[], int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void initialize_population(int **population, int population_size, int G) {
    for (int i = 0; i < population_size; i++) {
        for (int j = 0; j < G; j++) {
            population[i][j] = j + 1;
        }
        shuffle(population[i], G);
    }
}

double calculate_fitness(int *individual, int G, int **matrix1, int **matrix2) {
    double fitness = 0.0;
    for (int i = 0; i < G; i++) {
        for (int j = 0; j < G; j++) {
            fitness += matrix1[i][j] * matrix2[individual[i] - 1][individual[j] - 1];
        }
    }
    return fitness;
}

void partially_mapped_crossover(int *parent1, int *parent2, int *child, int G) {
    int point1 = rand() % G;
    int point2 = rand() % G;
    while (point1 == point2) {
        point2 = rand() % G;
    }

    if (point1 > point2) {
        int temp = point1;
        point1 = point2;
        point2 = temp;
    }

    for (int i = point1; i <= point2; i++) {
        child[i] = parent1[i];
    }

    for (int i = 0; i < G; i++) {
        if (i >= point1 && i <= point2) continue;

        int gene = parent2[i];
        while (1) {
            int found = 0;
            for (int j = point1; j <= point2; j++) {
                if (gene == parent1[j]) {
                    gene = parent2[j];
                    found = 1;
                    break;
                }
            }
            if (!found) {
                child[i] = gene;
                break;
            }
        }
    }
}

void exchange_mutation(int *individual, int G) {
    int point1 = rand() % G;
    int point2 = rand() % G;
    while (point1 == point2) {
        point2 = rand() % G;
    }
    int temp = individual[point1];
    individual[point1] = individual[point2];
    individual[point2] = temp;
}

void printMatrix(int **matrix, int size, const char *label) {
    printf("Matrix %s:\n", label);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}
 const char *filenames[] = {
    "bur26a.dat", "bur26b.dat", "bur26c.dat", "bur26d.dat",
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
    "tai60a.dat", "tai60b.dat", "tai64c.dat", "tai80a.dat",
    "tai80b.dat", "tai100a.dat", "tai100b.dat", "tai150b.dat",
    "tai256c.dat", "tho30.dat", "tho40.dat", "tho150.dat",
    "wil50.dat", "wil100.dat"
};
const double optimal[] = {
    5426670, 3817852, 5426795, 3821225,
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
    const double max_time[] = {
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120};


int main() {
    srand(time(NULL)); // Seed the random number generator

    int population_size = 100;
    int N_generation = 1000;
    double Pm = 0.15;
    int use_generation_limit; // 1 for using generation limit, 0 for using time limit

    printf("Parameters:\n");
    printf("Number of Generations: %d\n", N_generation);
    printf("Population Size: %d\n", population_size);
    printf("Mutation Probability: %0.2f\n", Pm);
    printf("Do you want to use generation limit? (1 for Yes, 0 for No): ");
    scanf("%d", &use_generation_limit);

 
 const char *filenames[] = {
    "bur26a.dat", "bur26b.dat", "bur26c.dat", "bur26d.dat",
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
    "lipa70a.dat", "lipa70b.dat", "lipa80a.dat", "ipa80b.dat",
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
    "tai60a.dat", "tai60b.dat", "tai64c.dat", "tai80a.dat",
    "tai80b.dat", "tai100a.dat", "tai100b.dat", "tai150b.dat",
    "tai256c.dat", "tho30.dat", "tho40.dat", "tho150.dat",
    "wil50.dat", "wil100.dat"
};
    const double optimal[] = {
    5426670, 3817852, 5426795, 3821225,
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
    const double max_time[] = {
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120, 120};

    int num_files = sizeof(filenames) / sizeof(filenames[0]);

    FILE *resultsFile = fopen("results.txt", "w");
    if (!resultsFile) {
        fprintf(stderr, "Error opening results file.\n");
        return 1; // Exit if file cannot be opened
    }

   fprintf(resultsFile, "Parameters: Number of Generations: %d, Population Size: %d, Mutation Probability: %0.2f\n\n", N_generation, population_size, Pm);

   for (int file_idx = 0; file_idx < num_files; file_idx++) {
        FILE *file = fopen(filenames[file_idx], "r");
        if (!file) {
            fprintf(stderr, "Error opening file %s\n", filenames[file_idx]);
            continue; // Skip to the next file
        }

        int **matrix1, **matrix2, G;
        extractMatrices(file, &matrix1, &matrix2, &G);
        fclose(file);

        printf("*********************************** %s ***********************************\n", filenames[file_idx]);

        int **PopulationP = (int **)malloc(population_size * sizeof(int *));
        int **PopulationW = (int **)malloc(population_size * sizeof(int *));
        for (int i = 0; i < population_size; i++) {
            PopulationP[i] = (int *)malloc(G * sizeof(int));
            PopulationW[i] = (int *)malloc(G * sizeof(int));
        }

        initialize_population(PopulationP, population_size, G);

        clock_t start_time = clock();
        double best_fitness = DBL_MAX;
        int best_index = -1;
        double time_spent = 0;

        int generation = 0;
        while ((use_generation_limit && generation < N_generation) || (!use_generation_limit && time_spent < max_time[file_idx])) {
            for (int i = 0; i < population_size; i++) {
                int partner_index = rand() % population_size;
                partially_mapped_crossover(PopulationP[i], PopulationP[partner_index], PopulationW[i], G);

                if ((double)rand() / RAND_MAX < Pm) {
                    exchange_mutation(PopulationW[i], G);
                }
            }

            for (int i = 0; i < population_size; i++) {
                double fitness_P = calculate_fitness(PopulationP[i], G, matrix1, matrix2);
                double fitness_W = calculate_fitness(PopulationW[i], G, matrix1, matrix2);
                if (fitness_W < fitness_P) {
                    for (int j = 0; j < G; j++) {
                        PopulationP[i][j] = PopulationW[i][j];
                    }
                }
                if (fitness_W < best_fitness) {
                    best_fitness = fitness_W;
                    best_index = i;
                }
            }


            generation++;
            clock_t current_time = clock();
            time_spent = (double)(current_time - start_time) / CLOCKS_PER_SEC;
            /*

            if (best_fitness <= optimal[file_idx]) {
                printf("Optimal %0.0f reached in %0.3f sec!, Optimal: %0.0f \n",best_fitness,time_spent, optimal[file_idx]);
                printf("Best Individual: [");
                for (int i = 0; i < G; i++) {
                    printf("%d", PopulationP[best_index][i]);
                    if (i < G - 1) printf(", ");
                }
                printf("]\n\n");
                //evaluate_specific_permutation(matrix1, matrix2, G, PopulationW[best_index]);
                break; // Optimal reached, break the loop
            }
            */
        }

        double gap = (best_fitness - optimal[file_idx]) / optimal[file_idx] * 100;
        printf("Best Fitness: %0.0f, Optimal: %0.0f ,Time: %0.3f sec, GAP: %f%%\n", best_fitness, optimal[file_idx],time_spent, gap);

        fprintf(resultsFile, "Instance: %s, Best Fitness: %0.0f, Gap: %0.2f%%\n", filenames[file_idx], best_fitness, gap);
        printf("Best Individual: [");
        for (int i = 0; i < G; i++) {
            printf("%d", PopulationP[best_index][i]);
            if (i < G - 1) printf(", ");
        }
        printf("]\n\n");


        // Free allocated memory
        for (int i = 0; i < population_size; i++) {
            free(PopulationP[i]);
            free(PopulationW[i]);
        }
        free(PopulationP);
        free(PopulationW);

        for (int i = 0; i < G; i++) {
            free(matrix1[i]);
            free(matrix2[i]);
        }
        free(matrix1);
        free(matrix2);
    }
    fclose(resultsFile);
    return 0;
}
