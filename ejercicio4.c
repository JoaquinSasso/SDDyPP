#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void medir_rendimiento(int N, int max_hilos) {
    // Asignación de memoria
    int *A = (int *)malloc(N * sizeof(int));
    int *B = (int *)malloc(N * sizeof(int));
    int *C = (int *)malloc(N * sizeof(int));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Error asignando memoria para N = %d\n", N);
        return;
    }

    // Inicialización de los vectores 
    // (Lo hacemos en paralelo para acelerar la preparación, pero no lo medimos)
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = N - i;
    }

    printf("\n========================================\n");
    printf("Prueba con N = %d componentes\n", N);
    printf("========================================\n");

    // 1. Ejecución Secuencial
    double inicio_sec = omp_get_wtime();
    
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
    
    double tiempo_sec = omp_get_wtime() - inicio_sec;
    printf("Secuencial: \t\t%f segundos\n", tiempo_sec);

    // 2. Ejecución Paralela variando hilos
    int num_hilos_a_probar[] = {2, 4, 8, 16};
    int cantidad_pruebas = sizeof(num_hilos_a_probar) / sizeof(num_hilos_a_probar[0]);

    for (int t = 0; t < cantidad_pruebas; t++) {
        int hilos = num_hilos_a_probar[t];
        
        // No probar más hilos de los que soporta tu procesador
        if (hilos > max_hilos) break; 

        omp_set_num_threads(hilos); // Definir la cantidad de hilos a usar
        
        double inicio_par = omp_get_wtime();
        
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            C[i] = A[i] + B[i];
        }
        
        double tiempo_par = omp_get_wtime() - inicio_par;
        
        // Cálculo del "Speedup" (Aceleración)
        double speedup = tiempo_sec / tiempo_par;
        
        printf("Paralelo (%2d hilos): \t%f segundos \t(Aceleración: %.2fx)\n", 
               hilos, tiempo_par, speedup);
    }

    // Prevenir que el compilador optimice y elimine el cálculo sumando un valor al azar
    if (C[N/2] == -1) printf("Ignorar esto.\n");

    free(A);
    free(B);
    free(C);
}

int main() {
    int max_hilos = omp_get_max_threads();
    printf("Hilos maximos soportados por tu CPU: %d\n", max_hilos);

    // Tamaños de los vectores a evaluar
    // 1 Millón, 10 Millones, 100 Millones
    int tamanos_N[] = {1000000, 10000000, 100000000};
    int num_tamanos = sizeof(tamanos_N) / sizeof(tamanos_N[0]);

    for(int i = 0; i < num_tamanos; i++) {
        medir_rendimiento(tamanos_N[i], max_hilos);
    }

    return 0;
}
