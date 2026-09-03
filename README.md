# Critical y Barrier en OpenMP — Ejercicios 8 y 9

Práctico de OpenMP · Sistemas Distribuidos y Paralelos 2026

Dos problemas distintos que se confunden todo el tiempo: **critical** resuelve *quién* toca qué, **barrier** resuelve *cuándo*. Abajo está el código, la salida real de cada ejecución y el razonamiento para defenderlo.

---

## La distinción que hay que tener clara

Todo lo que sigue sale de una sola observación: dentro de una región paralela, las variables **compartidas** (shared) viven en una única posición de memoria a la que llegan varios hilos al mismo tiempo. Eso genera exactamente dos tipos de problema, y OpenMP tiene una herramienta para cada uno.

| Directiva | Problema que resuelve | Qué hace exactamente |
|---|---|---|
| `critical` | Varios hilos **escriben la misma variable** a la vez y se pisan (condición de carrera). | Exclusión mutua: solo un hilo por vez ejecuta el bloque; los demás esperan turno. |
| `barrier` | Un hilo **avanza a la etapa siguiente** antes de que los otros terminen la anterior. | Sincronización: nadie pasa del punto hasta que *todos* los hilos del equipo llegan. |

**Regla mnemotécnica:** `critical` ordena el **acceso** (uno a la vez, en cualquier orden). `barrier` ordena el **tiempo** (todos juntos, en el mismo punto). Un programa puede necesitar las dos, o solo una.

---

## Ejercicio 8 · A — critical

### El problema que hay que mostrar primero

La línea `suma += 1` parece una sola operación, pero el procesador la ejecuta en tres pasos: **leer** el valor de memoria, **sumarle** uno en un registro, y **escribirlo** de vuelta. Si dos hilos leen el valor 500 al mismo tiempo, los dos calculan 501, y los dos escriben 501. Se hicieron dos incrementos y el contador avanzó uno solo: **se perdió una suma**.

Ese es el punto de todo el ejercicio. No alcanza con decir que `critical` sirve para proteger; hay que *producir el error* y después mostrarlo corregido.

```c
// ej8_critical.c
#include <stdio.h>
#include <omp.h>

#define N 1000000   /* cantidad de sumas: grande para que la carrera se note */

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
    printf("Sin critical      : %ld  %s\n", suma_sin_critical,
           (suma_sin_critical == esperado) ? "(salio bien por suerte)" : "<-- SE PERDIERON INCREMENTOS");
    printf("Con critical      : %ld  (siempre correcto)\n", suma_con_critical);

    return 0;
}
```

Salida real:

```
$ gcc -fopenmp -O0 ej8_critical.c -o ej8_critical && ./ej8_critical
Valor esperado    : 1000000
Sin critical      : 812223  <-- SE PERDIERON INCREMENTOS
Con critical      : 1000000  (siempre correcto)
```

Se perdieron casi 190.000 incrementos. Y el número **cambia en cada ejecución**: esa es la firma de una condición de carrera, y conviene correrlo dos o tres veces delante de la profesora para mostrarlo.

> **Detalle de compilación que te pueden preguntar:** se compila con `-O0` a propósito. Con optimización agresiva el compilador puede mantener la suma en un registro y cambiar el patrón del error. Con `-O0` la carrera aparece de forma consistente.

### Por qué no se usa critical siempre

Porque adentro del `critical` el programa deja de ser paralelo: los hilos hacen fila. En este ejemplo *todo* el cuerpo del bucle está protegido, así que la versión correcta es más lenta que la secuencial. Es el precio de la corrección, y hay que decirlo explícitamente.

- **La región crítica debe ser lo más chica posible**: solo la actualización de la variable compartida, nunca el cálculo que la precede.
- Para el caso concreto de acumular (`+ - * min max`), la solución correcta es `reduction(+:suma)`: cada hilo acumula en una copia privada y OpenMP combina los parciales al final, con una sola sincronización en vez de un millón.
- Para una única operación aritmética simple sobre memoria, `#pragma omp atomic` es más barato que `critical` porque usa instrucciones atómicas del hardware en lugar de un candado.

---

## Ejercicio 8 · B — barrier

### El problema es otro

Acá nadie escribe la misma posición de memoria: cada hilo escribe `v[tid]`, su propia casilla. No hay condición de carrera y `critical` no haría absolutamente nada. El problema es de **etapas**.

El algoritmo tiene dos fases. En la fase 1 cada hilo *escribe* su parte del vector. En la fase 2 cada hilo *lee el vector completo*. La fase 2 de un hilo depende de la fase 1 de todos los demás, y los hilos no avanzan al mismo ritmo: el hilo rápido llega a leer cuando el vector todavía está a medio llenar.

```c
// ej8_barrier.c
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
        int tid = omp_get_thread_num();   /* declarada adentro => private automatica */
        int suma = 0, j;

        /* FASE 1: cada hilo escribe SOLO su casilla.
           El usleep hace que los hilos terminen en momentos distintos. */
        usleep(tid * 100000);
        v[tid] = tid + 1;

        /* FASE 2: cada hilo necesita LEER TODO el vector.
           Como no hay punto de sincronizacion, el hilo 0 (el mas rapido)
           lee posiciones que los otros hilos todavia no escribieron. */
        for (j = 0; j < NH; j++) suma += v[j];

        printf("Hilo %d -> suma leida = %d (esperado %d)\n", tid, suma, esperado);
    }

    /* ================= CASO 2: CON barrier ================= */
    for (i = 0; i < NH; i++) v[i] = -1;

    printf("\n--- CON barrier ---\n");
    #pragma omp parallel num_threads(NH)
    {
        int tid = omp_get_thread_num();
        int suma = 0, j;

        usleep(tid * 100000);
        v[tid] = tid + 1;                 /* FASE 1 */

        /* Punto de encuentro: cada hilo se frena aca hasta que LLEGAN TODOS.
           Recien entonces el vector esta completo y es seguro leerlo.
           IMPORTANTE: todos los hilos del equipo deben alcanzar el barrier;
           si lo pusieramos dentro de un 'if (tid==0)' el programa se cuelga. */
        #pragma omp barrier

        for (j = 0; j < NH; j++) suma += v[j];   /* FASE 2 */

        printf("Hilo %d -> suma leida = %d (esperado %d)\n", tid, suma, esperado);
    }

    return 0;
}
```

Salida real:

```
$ gcc -fopenmp ej8_barrier.c -o ej8_barrier && ./ej8_barrier
--- SIN barrier ---
Hilo 0 -> suma leida = -2 (esperado 10)
Hilo 1 -> suma leida = 1  (esperado 10)
Hilo 2 -> suma leida = 5  (esperado 10)
Hilo 3 -> suma leida = 10 (esperado 10)

--- CON barrier ---
Hilo 3 -> suma leida = 10 (esperado 10)
Hilo 1 -> suma leida = 10 (esperado 10)
Hilo 2 -> suma leida = 10 (esperado 10)
Hilo 0 -> suma leida = 10 (esperado 10)
```

Fijate en la salida sin barrier: el hilo 0 lee `1 + (-1) + (-1) + (-1) = -2`, el hilo 1 lee `1 + 2 - 1 - 1 = 1`, el hilo 2 lee `1+2+3-1 = 5`. Cada hilo ve exactamente el estado del vector en el instante en que pasó. El único que acierta es el último. Con barrier los cuatro dan 10.

> **El detalle que hace que el ejemplo se vea:** el `usleep(tid * 100000)` no forma parte del algoritmo: obliga a que los hilos se desincronicen de manera visible. Sin él el error igual existiría, pero aparecería una de cada tantas ejecuciones y sería difícil de mostrar. Conviene decirlo antes de que te lo pregunten.

**Dos errores clásicos con barrier:**

- **Deadlock:** si el `barrier` queda dentro de un `if (tid == 0)`, de un `single` o de una rama que no todos recorren, los hilos que llegan esperan para siempre a los que nunca van a llegar. Todos los hilos del equipo tienen que alcanzarlo.
- **Barrier redundante:** `for`, `sections`, `single` y el cierre de la región `parallel` ya tienen un *barrier implícito*. Escribir uno a mano ahí no hace nada. Solo se elimina con `nowait`.

---

## Ejercicio 9 — análisis del código dado

Es una suma de vectores, `c[i] = a[i] + b[i]`, con N = 100. Lo que importa no es el cálculo sino la construcción: primero se crea el equipo de hilos con `parallel`, y recién adentro se reparte el trabajo con `for`.

```c
// ej9.c — el código del práctico, comentado
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
```

### Cláusula por cláusula

| Elemento | Qué hace y por qué está |
|---|---|
| `#pragma omp parallel` | Marca el **fork**: crea el equipo de hilos. Todo lo que está en el bloque lo ejecuta cada hilo *completo*, salvo lo que esté dentro de una directiva de reparto de trabajo. |
| `shared(a, b, c)` | Los tres vectores son una única copia en memoria. Es lo que se quiere: cada hilo lee su porción de `a` y `b` y escribe su porción de `c`. No hay carrera porque cada índice `i` lo procesa un solo hilo. |
| `shared(nthreads)` | La escribe el hilo 0 para reportar el tamaño del equipo. Al ser compartida, el valor queda disponible fuera del bloque. |
| `shared(chunk)` | Es un parámetro de configuración: todos los hilos leen el mismo 10. Solo se lee, nunca se modifica. |
| `private(tid)` | **Obligatorio.** Cada hilo tiene su propio identificador. Si fuera shared, los hilos se pisarían el valor y todos imprimirían el mismo id. |
| `private(i)` | Cada hilo recorre su propio rango de iteraciones, así que necesita su propio contador. Nota: la variable de control de un `omp for` es privada por defecto aunque no se declare; acá está explícito porque `i` se declaró fuera y se usa también en el bucle secuencial. |
| `omp_get_thread_num()` | Devuelve el id del hilo que la llama, de 0 a nthreads−1. El 0 es el hilo maestro. |
| `omp_get_num_threads()` | Devuelve la cantidad de hilos del equipo actual. **Si se la llamara fuera de la región paralela devolvería 1**: por eso está adentro. |
| `#pragma omp for` | Reparte las 100 iteraciones entre los hilos. Sin esta directiva, cada hilo ejecutaría las 100 y se harían 400 sumas repetidas. |
| `schedule(dynamic, 10)` | Divide el bucle en bloques de 10 iteraciones. Cada hilo agarra un bloque, lo procesa y vuelve a pedir el siguiente que esté libre. El reparto se decide en ejecución, no en compilación. |

### Salida y cómo interpretarla

```
$ OMP_NUM_THREADS=4 ./ej9   (fragmento)
Thread 2 starting...
Thread 2: c[0]= 0.000000
Thread 2: c[1]= 2.000000
   ... hasta c[9] — su primer bloque de 10
Number of threads = 4
Thread 0 starting...
Thread 0: c[20]= 40.000000
Thread 0: c[21]= 42.000000
   ...
```

Tres cosas para señalar en la defensa:

1. **La salida sale desordenada y cambia en cada corrida.** Los `printf` de distintos hilos se intercalan; incluso el `Number of threads` puede aparecer después de que otro hilo ya empezó. Eso no es un error del programa, es que el orden de ejecución de los hilos no está determinado.
2. **Los índices vienen de a bloques de 10 por hilo** (0–9, 10–19, 20–29…). Esa es la huella visible del `chunk = 10`.
3. **Las 100 iteraciones se hacen una sola vez en total**, no 100 por hilo. Se comprueba contando las líneas `c[...]`: son exactamente 100.

### La prueba de que el reparto es dinámico

Contando cuántas iteraciones le tocaron a cada hilo en tres ejecuciones seguidas del mismo binario:

```
$ for k in 1 2 3; do ./ej9 | grep "c\[" | awk '{print $2}' | sort | uniq -c; done

-- Ejecucion 1 --      -- Ejecucion 2 --      -- Ejecucion 3 --
  20 Hilo 0             100 Hilo 2              20 Hilo 0
  10 Hilo 1                                     50 Hilo 1
  10 Hilo 2                                     30 Hilo 3
  60 Hilo 3
```

El reparto es distinto cada vez, y en la ejecución 2 un solo hilo se llevó los 100. Eso es `dynamic` funcionando como corresponde: el trabajo va al hilo que lo pide, y si un hilo se despierta antes y el cuerpo del bucle es trivial, se lleva todos los bloques antes de que los otros arranquen. Con `schedule(static, 10)` el reparto sería siempre el mismo, fijado antes de empezar.

> **La crítica que conviene adelantar:** para este bucle `dynamic` es una mala elección. Todas las iteraciones cuestan exactamente lo mismo (una suma), así que el reparto dinámico solo agrega la sobrecarga de pedir bloques sin ningún beneficio; `static` sería más rápido. `dynamic` se justifica cuando las iteraciones tienen costo desigual — por ejemplo, buscar divisores de `i`, donde los `i` grandes tardan mucho más. Además el `printf` adentro del bucle domina el tiempo de ejecución y serializa la salida por consola: este código sirve para ver el reparto, no para medir rendimiento.

---

## Preguntas probables en la defensa

**¿Por qué en el ejercicio 9 no hace falta critical si `c` es shared?**
Porque cada iteración `i` escribe únicamente `c[i]`, y el `omp for` garantiza que cada `i` se lo asigna a un solo hilo. Dos hilos nunca tocan la misma posición. `critical` haría falta si todos escribieran una misma variable, como en el ejercicio 8.

**¿Qué pasa si saco `private(tid)`?**
`tid` pasa a ser compartida: los cuatro hilos escriben la misma posición de memoria y cada uno lee lo que dejó el último. Los `printf` muestran ids repetidos o inconsistentes, y la condición `if (tid == 0)` puede cumplirla un hilo que no es el maestro, o ninguno.

**¿Cuál es la diferencia entre `parallel` y `parallel for`?**
`parallel` crea el equipo y hace que *todos* ejecuten el mismo bloque completo. `for` reparte iteraciones entre hilos ya existentes. `parallel for` es la abreviatura de las dos juntas. En el ejercicio 9 están separadas justamente porque hay código previo (los `printf` de presentación) que sí tiene que ejecutar cada hilo.

**¿Dónde hay sincronización en el ejercicio 9, si no escribí ningún `barrier`?**
Hay dos barriers implícitos: uno al final del `omp for` y otro al cerrar la región `parallel` (el join). El primero se podría eliminar con `nowait`; el del cierre de la región no.

**¿`critical` o `atomic`?**
`atomic` sirve solo para una operación aritmética simple sobre una posición de memoria y la resuelve con instrucciones del procesador, así que es más barata. `critical` protege un bloque arbitrario de código con un candado. Para `suma += 1` alcanza `atomic`; para varias líneas que deben ejecutarse juntas hace falta `critical`.

**¿Cómo demostrás que hay condición de carrera y no un error de tu código?**
Corriendo el mismo binario varias veces sin recompilar: el resultado incorrecto cambia de valor en cada ejecución, y con un solo hilo (`OMP_NUM_THREADS=1`) da siempre el valor correcto. Un error de lógica daría siempre el mismo resultado equivocado.

---

*Compilación de los tres programas:* `gcc -fopenmp -O0 ej8_critical.c -o ej8_critical` · `gcc -fopenmp ej8_barrier.c -o ej8_barrier` · `gcc -fopenmp ej9.c -o ej9`. *La cantidad de hilos se fija con* `omp_set_num_threads()`, *con la cláusula* `num_threads()` *o con la variable de entorno* `OMP_NUM_THREADS`.