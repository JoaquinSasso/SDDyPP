#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void procesar_vector(int N) {
    printf("\n==================================================\n");
    printf("Procesando vector de dimension N = %d\n", N);
    printf("==================================================\n");

    // Asignación de memoria y llenado del vector con números aleatorios (0 a 99)
    double *vec = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        vec[i] = (double)(rand() % 100);
    }

    double min_val = vec[0];
    double max_val = vec[0];
    double suma = 0.0;
    double promedio = 0.0;

    // Región paralela
    #pragma omp parallel
    {
        // El hilo maestro muestra la cantidad total de hilos en el equipo
        #pragma omp master
        {
            printf("[Hilo Maestro ID %d]: Cantidad total de hilos en ejecucion = %d\n\n", 
                   omp_get_thread_num(), omp_get_num_threads());
        }

        // Definición de las secciones paralelas
        #pragma omp sections
        {
            // Sección 1: Cálculo del Máximo
            #pragma omp section
            {
                int id_hilo = omp_get_thread_num();
                printf(" -> [Seccion 1 - Maximo] procesada por el hilo ID: %d\n", id_hilo);
                
                double max_local = vec[0];
                for (int i = 1; i < N; i++) {
                    if (vec[i] > max_local) {
                        max_local = vec[i];
                    }
                }
                max_val = max_local;
            }

            // Sección 2: Cálculo del Mínimo
            #pragma omp section
            {
                int id_hilo = omp_get_thread_num();
                printf(" -> [Seccion 2 - Minimo] procesada por el hilo ID: %d\n", id_hilo);
                
                double min_local = vec[0];
                for (int i = 1; i < N; i++) {
                    if (vec[i] < min_local) {
                        min_local = vec[i];
                    }
                }
                min_val = min_local;
            }

            // Sección 3: Cálculo del Promedio
            #pragma omp section
            {
                int id_hilo = omp_get_thread_num();
                printf(" -> [Seccion 3 - Promedio] procesada por el hilo ID: %d\n", id_hilo);
                
                double suma_local = 0.0;
                for (int i = 0; i < N; i++) {
                    suma_local += vec[i];
                }
                suma = suma_local;
            }
        }
    } // Fin de la región paralela (hay una barrera implícita aquí)

    // El cálculo final del promedio se hace de forma secuencial una vez obtenida la suma
    promedio = suma / N;

    printf("\nResultados para N = %d:\n", N);
    printf("Minimo:   %.2f\n", min_val);
    printf("Maximo:   %.2f\n", max_val);
    printf("Promedio: %.2f\n", promedio);

    free(vec);
}

int main() {
    // Semilla para generar números aleatorios
    srand(time(NULL));

    // Configuramos explícitamente 3 hilos para que cada sección tenga el suyo
    omp_set_num_threads(3);

    // Tamaños requeridos por el ejercicio
    int tamaños[] = {10000, 100000, 500000};

    // Procesamos cada tamaño
    for (int i = 0; i < 3; i++) {
        procesar_vector(tamaños[i]);
    }

    return 0;
}
