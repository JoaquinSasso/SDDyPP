/* ============================================================
   Ejercicio 9 - Suma de vectores con parallel + for schedule(dynamic)
   ------------------------------------------------------------
   Compilar:  gcc -fopenmp ejercicio9.c -o ejercicio9
   Ejecutar:  ./ejercicio9
   Compilar y ejecutar: gcc -fopenmp ejercicio9.c -o ejercicio9 && ./ejercicio9
   ============================================================ */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define CHUNKSIZE 10    /* tamano del bloque de iteraciones que toma cada hilo */
#define N 100           /* dimension de los vectores */

int main(int argc, char *argv[])
{
    int nthreads, tid, i, chunk;
    float a[N], b[N], c[N];

    /* Inicializacion SECUENCIAL (fuera de la region paralela) */
    for (i = 0; i < N; i++)
        a[i] = b[i] = i * 1.0;

    chunk = CHUNKSIZE;

    /* --------------------------------------------------------
       Region paralela: aca se hace el FORK (se crea el equipo de hilos).
         shared(a,b,c,nthreads,chunk) -> una sola copia en memoria,
             visible para todos los hilos (los vectores y el chunk).
         private(i,tid)              -> cada hilo tiene SU propia copia,
             sin inicializar; imprescindible porque tid es distinto por
             hilo e i es el indice que cada hilo maneja por separado.
       -------------------------------------------------------- */
    #pragma omp parallel shared(a, b, c, nthreads, chunk) private(i, tid)
    {
        tid = omp_get_thread_num();       /* id del hilo actual: 0..nthreads-1 */

        if (tid == 0) {                   /* solo el hilo maestro informa */
            nthreads = omp_get_num_threads();   /* cuantos hilos hay en el equipo */
            printf("Number of threads = %d\n", nthreads);
        }

        printf("Thread %d starting...\n", tid);   /* lo ejecutan TODOS los hilos */

        /* ----------------------------------------------------
           Work-sharing: las N iteraciones NO se repiten en cada hilo,
           se REPARTEN entre los hilos del equipo.
           schedule(dynamic, chunk): cada hilo toma un bloque de 10
           iteraciones; al terminarlo pide el siguiente bloque libre.
           El reparto se decide en tiempo de ejecucion -> util cuando
           las iteraciones no cuestan todas lo mismo.
           Al final del for hay un BARRIER implicito.
           ---------------------------------------------------- */
        #pragma omp for schedule(dynamic, chunk)
        for (i = 0; i < N; i++) {
            c[i] = a[i] + b[i];           /* cada i toca una posicion distinta:
                                             no hay condicion de carrera */
            printf("Thread %d: c[%d]= %f\n", tid, i, c[i]);
        }

    }   /* fin de la region paralela: JOIN, los hilos se sincronizan y mueren */

    return 0;
}