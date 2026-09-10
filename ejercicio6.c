//Ejercicio 6 Realizar un código paralelo donde se encuentre el máximo y el mínimo
//de un vector de dimensión N= 10000, 100000, 5000000, utilizando 4, 8 y 32 hilos.
//Tome el tiempo de cada una de las ejecuciones.

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <omp.h>

void generarVector(int num, int v[]){
    for(int i = 0; i < num; i++){
        v[i] = rand() % 100000000;
    }
}

void ejecutar (int hilos, int num, int v[]){
    omp_set_num_threads(hilos);
    int max = 0;
    int min = INT_MAX;
    double inicio = omp_get_wtime();

    #pragma omp parallel for reduction(max:max) reduction(min:min)
    for (int i = 0; i < num; i++){
        if(v[i] > max){
            max = v[i];
        }
        if(v[i] < min){
            min = v[i];
        }
    }
    printf("Maximo: %d\n", max);
    printf("Minimo: %d\n", min);
    double fin = omp_get_wtime();
    printf("Tiempo: %f\n", fin - inicio);
}

int main(){
    srand(time(NULL));
    int hilos[] = {4, 8, 32};
    int numeros[] = {10000, 100000, 5000000};
    int *v1 = malloc(numeros[0] * sizeof(int));
    int *v2 = malloc(numeros[1] * sizeof(int));
    int *v3 = malloc(numeros[2] * sizeof(int));

    generarVector(numeros[0], v1);
    for (int i = 0; i < 3; i++){
        printf("%d hilos, %d elementos\n", hilos[i], numeros[0]);
        ejecutar(hilos[i], numeros[0], v1);
        printf("-------\n");
    }
    free(v1);

    generarVector(numeros[1], v2);
    for (int i = 0; i < 3; i++){
        printf("%d hilos, %d elementos\n", hilos[i], numeros[1]);
        ejecutar(hilos[i], numeros[1], v2);
        printf("-------\n");
    }
    free(v2);

    generarVector(numeros[2], v3);
    for (int i = 0; i < 3; i++){
        printf("%d hilos, %d elementos\n", hilos[i], numeros[2]);
        ejecutar(hilos[i], numeros[2], v3);
        printf("-------\n");
    }
    free(v3);
    return 0;
}