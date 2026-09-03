//gcc -fopenmp -o ejercicio1.exe ejercicio1.c && ./ejercicio1.exe
#include <stdio.h>
#include <omp.h>

int main()
{
   int numProcesadores, numHilosActuales, maxCanthilos, anidado;
   numProcesadores = omp_get_num_procs();
   printf("El numero de procesadores disponibles es: %d\n", numProcesadores);

   numHilosActuales = omp_get_num_threads();
   printf("El numero de hilos se estan utilizando actualmente es: %d\n", numHilosActuales);

   maxCanthilos = omp_get_max_threads();
   printf("El maximo numero de hilos disponibles es: %d\n", maxCanthilos);

   anidado = omp_get_nested();
   printf("Paralelismo anidado disponible: %s\n", omp_get_nested() ? "SI" : "NO");

   return 0;
}