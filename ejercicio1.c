/*Ejercicio 1 Realice un programa que muestre/verifique lo siguiente:
- Número de procesadores disponibles.
- Número de hilos que se están utilizando.
- Máximo número disponible de hilos.
- Si el paralelismo anidado está disponible.
- Manipule la generación de varios threads y modificación del número de threads
*/

#include <stdio.h>
#include <omp.h>


int main() {
    int cant_procesadores, num_hilos_usando, num_max_hilos, paralelismo_anidado; 
    int cant_hilos = 2;
    int thread_id = omp_get_thread_num();
    #pragma omp parallel num_threads(cant_hilos)
    {
        cant_procesadores = omp_get_num_procs();
        num_hilos_usando = omp_get_num_threads();
        num_max_hilos = omp_get_max_threads();
        paralelismo_anidado = omp_get_nested();

    }
    printf("Numero de procesadores disponibles: %d, hilo usado: %d\n", cant_procesadores, thread_id);
    printf("Numero de hilos que se estan utilizando: %d\n", num_hilos_usando);
    printf("Maximo numero disponible de hilos: %d\n", num_max_hilos);
    printf("Paralelismo anidado disponible: %d\n", paralelismo_anidado);
    return 0;
}
 // como tomar el tiempo y hacer programa en paralelo con openmp
