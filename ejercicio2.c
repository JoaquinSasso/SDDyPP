/* Ejercicio 2 ¿Cuál es la función de la cláusula private?. Realice un ejemplo
explicando cómo influye la presencia y ausencia de esta cláusula. Investigue cómo
se comparten los datos entre los distintos hilos.

- Función de la cláusula private: Crea una copia privada e independiente de una variable para cada hilo en su propia pila de memoria. No hereda el valor inicial ni modifica el original al salir.
- Presencia vs Ausencia: 
  * Sin private (shared): Todos los hilos leen y escriben en la misma memoria (Condición de carrera / Data race).
  * Con private: Cada hilo trabaja sobre su propia copia aislada sin interferir con otros hilos.
- Compartición de datos en OpenMP (Memoria compartida):
  * shared: Memoria común entre todos los hilos.
  * private: Copia privada sin inicializar para cada hilo.
  * firstprivate: Copia privada inicializada con el valor previo.
  * lastprivate: Copia privada que devuelve al final el valor de la última iteración.
  * reduction: Copia local neutra que combina los resultados de manera segura al terminar.
*/

/* ============================================================
   Compilar:  gcc -fopenmp ejercicio2.c -o ejercicio2
   Ejecutar:  ./ejercicio2
   Compilar y ejecutar: gcc -fopenmp ejercicio2.c -o ejercicio2 && ./ejercicio2
   ============================================================ */

#include <stdio.h>
#include <omp.h>

int main() {
    int valor = 10;

    // 1. SIN private (ausencia: variable compartida por defecto)
    printf("=== SIN PRIVATE (Variable compartida) ===\n");
    #pragma omp parallel num_threads(8)
    {
        valor = omp_get_thread_num();
        printf("Hilo %d ve  su valor local = %d\n", omp_get_thread_num(), valor);
    }
    printf("Valor final fuera del bloque: %d\n\n", valor);

    // Resetear valor
    valor = 10;

    // 2. CON private (presencia: copia propia por hilo)
    printf("=== CON PRIVATE (Variable privada) ===\n");
    #pragma omp parallel num_threads(8) private(valor)
    {
        valor = omp_get_thread_num();
        printf("Hilo %d ve su valor local = %d\n", omp_get_thread_num(), valor);
    }
    printf("Valor final fuera del bloque: %d\n", valor);

    return 0;
}
