/* ============================================================
   Compilar:  gcc -fopenmp ejercicio7.c -o ejercicio7
   Ejecutar:  ./ejercicio7
   Compilar y ejecutar: gcc -fopenmp ejercicio7.c -o ejercicio7 && ./ejercicio7
   ============================================================ */
//Ejercicio 7 Genere un programa paralelo que permita buscar un valor en un vector
//desordenado de dimensión N= 400000, 1000000. Indique la cantidad de hilos en su
//procesador y encuentre la cantidad de hilos que minimizan el tiempo de ejecución.
//(Pruebe con 4, 8, 16 y 64 hilos)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void generar(int num, int v[]){
	for (int i = 0; i < num; i++){
		v[i] = rand() % 1000000000;
	}
}

void buscar(int hilos, int numV, int v[], int num){
	omp_set_num_threads(hilos);
	int encontrado = numV;
	double inicio = omp_get_wtime();
	#pragma omp parallel for reduction(min:encontrado)
	for (int i = 0; i < numV; i++){
		if (v[i] == num){
			encontrado = i;
		}
	}
	double fin = omp_get_wtime();
	if (encontrado != numV){
		printf("Numero encontrado en la posicion %d\n", encontrado);
	} else {
		printf("Numero no encontrado\n");
	}
	printf("Tiempo de ejecucion: %f segundos\n", fin - inicio);
}

int main(){
	srand(time(NULL));
	int hilos[] = {4, 8, 16, 64};
	int numV1 = 400000, numV2 = 1000000;
	int *v1 = malloc(numV1 * sizeof(int));
	int *v2 = malloc(numV2 * sizeof(int));


	printf("Maxima cantidad de hilos: %d\n-------\n", omp_get_max_threads());

	generar(numV1, v1);
	int num1 = v1[rand() % numV1];
	printf("Numero a buscar: %d\n-------\n", num1);
	for (int i = 0; i < 4; i++){
		printf("%d hilos, 400000 elementos\n", hilos[i]);
		buscar(hilos[i], numV1, v1, num1);
		printf("-------\n");
	}
	free(v1);

	generar(numV2, v2);
	int num2 = v2[rand() % numV2];
	printf("Numero a buscar: %d\n-------\n", num2);
	for (int i = 0; i < 4; i++){
		printf("%d hilos, 1000000 elementos\n", hilos[i]);
		buscar(hilos[i], numV2, v2, num2);
		printf("-------\n");
	}
	free(v2);
	return 0;
}