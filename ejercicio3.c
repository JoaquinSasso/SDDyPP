/* Ejercicio 3: Investigue la cláusula schedule y los distintos tipos de reparto de
   iteraciones. Genere un programa paralelo que permita contar la cantidad de
   números menores a 500 en un vector de dimensión N utilizando las distintas
   políticas de schedule. ¿Cuál resulta más eficiente? Justifique.

   Resumen Teórico:
   - static: Reparto fijo al inicio. Ideal para cargas homogéneas.
     Ejemplo: Suma de vectores, conteos simples.

   - dynamic: Reparto a demanda en tiempo de ejecución. Ideal para cargas heterogéneas/impredecibles.
     Ejemplo: Test de primalidad / Matrices esparcidas.
     Explicación: Probar números primos grandes lleva miles de cálculos y pequeños muy poco;
     dynamic evita que unos hilos queden atrapados mientras otros están ociosos.

   - guided: Bloques decrecientes exponencialmente. Balancea carga reduciendo la sobrecarga de asignación.
     Ejemplo: Renderizado 3D / Mandelbrot / Ray Tracing.
     Explicación: Da bloques grandes al inicio para partes simples (fondos) y pequeños al final
     para repartir los píxeles complejos entre los hilos sin saturar con solicitudes.

   Justificación de Examen:
   Static es la política más eficiente para este problema porque la carga por iteración es totalmente
   homogénea (evaluar vector[i] < 500) y evita la sobrecarga (overhead) de sincronización dinámica.
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100000000

int main() {
    int *vector = (int *)malloc(N * sizeof(int));
    if (vector == NULL) return 1;

    // Inicialización del vector
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        vector[i] = (i * 31 + 17) % 1000;
    }

    // Warm-up de la memoria cache
    long dummy = 0;
    #pragma omp parallel for schedule(static) reduction(+:dummy)
    for (int i = 0; i < N; i++) dummy += vector[i];

    double start;
    long count;

    // 1. Static (por defecto divide N / num_threads de forma equitativa)
    count = 0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static) reduction(+:count)
    for (int i = 0; i < N; i++) {
        if (vector[i] < 500) count++;
    }
    printf("Static : Menores a 500: %ld | Tiempo: %.6f s\n", count, omp_get_wtime() - start);

    // 2. Dynamic
    count = 0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 10000) reduction(+:count)
    for (int i = 0; i < N; i++) {
        if (vector[i] < 500) count++;
    }
    printf("Dynamic: Menores a 500: %ld | Tiempo: %.6f s\n", count, omp_get_wtime() - start);

    // 3. Guided
    count = 0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(guided, 10000) reduction(+:count)
    for (int i = 0; i < N; i++) {
        if (vector[i] < 500) count++;
    }
    printf("Guided : Menores a 500: %ld | Tiempo: %.6f s\n", count, omp_get_wtime() - start);

    free(vector);
    return 0;
}