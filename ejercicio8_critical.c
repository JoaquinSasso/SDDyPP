/* ============================================================
   Ejercicio 8 - Parte A: clausula/directiva CRITICAL
   ------------------------------------------------------------
   Objetivo: mostrar que una variable COMPARTIDA modificada por
   varios hilos a la vez se corrompe (condicion de carrera), y
   que #pragma omp critical lo soluciona serializando el acceso.

   Compilar:  gcc -fopenmp -O0 ejercicio8_critical.c -o ejercicio8_critical
   Ejecutar:  ./ejercicio8_critical
   Compilar y ejecutar: gcc -fopenmp -O0 ejercicio8_critical.c -o ejercicio8_critical && ./ejercicio8_critical

   Se compila con -O0 a propósito. Con optimización agresiva el compilador 
   puede mantener la suma en un registro y cambiar el patrón del error. 
   Con -O0 la carrera aparece de forma consistente.
   ============================================================ */

#include <stdio.h>
#include <omp.h>

#define N 1000000   /* cantidad de sumas (1 millon): grande para que la carrera se note */

int main(void)
{
    long suma_sin_critical = 0;   /* variable compartida SIN proteger */
    long suma_con_critical = 0;   /* variable compartida SI protegida  */
    long esperado = N;            /* cada iteracion suma 1 -> total = N */
    int i;

    omp_set_num_threads(4);       /* fijamos 4 hilos para que sea reproducible */

    /* ---------- CASO 1: SIN critical (INCORRECTO) ---------- */
    /* suma_sin_critical es shared por defecto (esta declarada fuera).
       'suma += 1' NO es atomico: son 3 pasos (leer, sumar, escribir).
       Dos hilos pueden leer el mismo valor viejo y una escritura pisa
       a la otra -> se pierden incrementos. */
    #pragma omp parallel for private(i)
    for (i = 0; i < N; i++) {
        suma_sin_critical += 1;   /* <-- CONDICION DE CARRERA */
    }

    /* ---------- CASO 2: CON critical (CORRECTO) ---------- */
    /* La region critical es un candado (exclusion mutua): OpenMP garantiza
       que a lo sumo UN hilo por vez ejecuta el bloque. Los demas esperan.
       El resultado es siempre exacto, pero se paga en tiempo. */
    #pragma omp parallel for private(i)
    for (i = 0; i < N; i++) {
        #pragma omp critical
        {
            suma_con_critical += 1;   /* acceso serializado -> seguro */
        }
    }

    printf("Valor esperado    : %ld\n", esperado);
    printf("Sin critical      : %ld  %s", suma_sin_critical,
           (suma_sin_critical == esperado) ? "(salio bien por suerte)\n" : " ");
   if(suma_sin_critical != esperado) printf("<-- SE PERDIERON %ld INCREMENTOS\n", esperado - suma_sin_critical);
    printf("Con critical      : %ld  (siempre correcto)\n", suma_con_critical);

    return 0;
}