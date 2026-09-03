/* ============================================================
   Ejercicio 8 - Parte B: directiva BARRIER
   ------------------------------------------------------------
   Objetivo: mostrar un caso donde la FASE 2 depende de que TODOS
   los hilos hayan terminado la FASE 1. Sin barrier los hilos
   rapidos leen datos que todavia no fueron escritos; con barrier
   nadie avanza hasta que llegan todos.

   Compilar:  gcc -fopenmp ej8_barrier.c -o ej8_barrier
   Ejecutar:  ./ej8_barrier
   Compilar y ejecutar: gcc -fopenmp ej8_barrier.c -o ej8_barrier && ./ej8_barrier
   ============================================================ */

#include <stdio.h>
#include <unistd.h>   /* usleep: solo para simular que unos hilos tardan mas */
#include <omp.h>

#define NH 4          /* cantidad de hilos = cantidad de casillas del vector */

int main(void)
{
    int v[NH];        /* vector COMPARTIDO: cada hilo escribe una posicion */
    int i, esperado = 0;

    for (i = 0; i < NH; i++) esperado += (i + 1);   /* 1+2+3+4 = 10 */

    /* ================= CASO 1: SIN barrier ================= */
    for (i = 0; i < NH; i++) v[i] = -1;             /* -1 = "todavia sin escribir" */

    printf("--- SIN barrier ---\n");
    #pragma omp parallel num_threads(NH)
    {
        int numHilo = omp_get_thread_num();   /* declarada adentro => private automatica */
        int suma = 0, j;

        /* FASE 1: cada hilo escribe SOLO su casilla.
           El usleep hace que los hilos terminen en momentos distintos. */
        usleep(numHilo * 100000);
        v[numHilo] = numHilo + 1;

        /* FASE 2: cada hilo necesita LEER TODO el vector.
           Como no hay punto de sincronizacion, el hilo 0 (el mas rapido)
           lee posiciones que los otros hilos todavia no escribieron. */
        for (j = 0; j < NH; j++) suma += v[j];

        printf("Hilo %d -> suma leida = %d (esperado %d)\n", numHilo, suma, esperado);
    }

    /* ================= CASO 2: CON barrier ================= */
    for (i = 0; i < NH; i++) v[i] = -1;

    printf("\n--- CON barrier ---\n");
    #pragma omp parallel num_threads(NH)
    {
        int numHilo = omp_get_thread_num();
        int suma = 0, j;

        usleep(numHilo * 100000);
        v[numHilo] = numHilo + 1;                 /* FASE 1 */

        /* Punto de encuentro: cada hilo se frena aca hasta que LLEGAN TODOS.
           Recien entonces el vector esta completo y es seguro leerlo.
           IMPORTANTE: todos los hilos del equipo deben alcanzar el barrier;
           si lo pusieramos dentro de un 'if (numHilo==0)' el programa se cuelga. */
        #pragma omp barrier

        for (j = 0; j < NH; j++) suma += v[j];   /* FASE 2 */

        printf("Hilo %d -> suma leida = %d (esperado %d)\n", numHilo, suma, esperado);
    }

    return 0;
}