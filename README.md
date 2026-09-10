# Práctico de OpenMP — Sistemas Distribuidos, Descentralizados y Paralelos 2026

## Tabla de Contenidos

- [Introducción a OpenMP](#introducción-a-openmp)
- [Requisitos y Compilación](#requisitos-y-compilación)
- [Conceptos Clave de OpenMP](#conceptos-clave-de-openmp)
  - [Modelo Fork-Join](#modelo-fork-join)
  - [Serial vs Paralelo: cuándo conviene paralelizar](#serial-vs-paralelo-cuándo-conviene-paralelizar)
  - [Directivas principales](#directivas-principales)
  - [Cláusulas de datos: shared, private, firstprivate, lastprivate](#cláusulas-de-datos)
  - [Sincronización: critical, barrier, atomic](#sincronización-critical-barrier-atomic)
  - [Reduction](#reduction)
  - [Schedule](#schedule)
  - [Funciones de runtime](#funciones-de-runtime)
- [Ejercicio 1 — Info del entorno OpenMP](#ejercicio-1--info-del-entorno-openmp)
- [Ejercicio 2 — Cláusula private vs shared](#ejercicio-2--cláusula-private-vs-shared)
- [Ejercicio 3 — Schedule y conteo paralelo](#ejercicio-3--schedule-y-conteo-paralelo)
- [Ejercicio 4 — Suma de vectores con reduction y comparación de hilos](#ejercicio-4--suma-de-vectores-con-reduction-y-comparación-de-hilos)
- [Ejercicio 5 — Sections: mínimo, máximo y promedio](#ejercicio-5--sections-mínimo-máximo-y-promedio)
- [Ejercicio 6 — Máximo y mínimo con reduction(max/min)](#ejercicio-6--máximo-y-mínimo-con-reductionmaxmin)
- [Ejercicio 7 — Búsqueda paralela en vector desordenado](#ejercicio-7--búsqueda-paralela-en-vector-desordenado)
- [Ejercicio 8 — critical y barrier](#ejercicio-8--critical-y-barrier)
  - [8A — critical: condición de carrera](#8a--critical-condición-de-carrera)
  - [8B — barrier: sincronización entre fases](#8b--barrier-sincronización-entre-fases)
- [Ejercicio 9 — Análisis de código: parallel + for + schedule(dynamic)](#ejercicio-9--análisis-de-código-parallel--for--scheduledynamic)
- [example.c — Ejemplo mínimo de reduction](#examplec--ejemplo-mínimo-de-reduction)
- [Resumen para la defensa](#resumen-para-la-defensa)

---

## Introducción a OpenMP

**OpenMP** (Open Multi-Processing) es una API para programación paralela en **memoria compartida**. Consiste en:

- **Directivas de compilador** (`#pragma omp ...`): le indican al compilador qué regiones paralelizar.
- **Funciones de biblioteca** (`omp_get_thread_num()`, etc.): permiten consultar y controlar el entorno de ejecución.
- **Variables de entorno** (`OMP_NUM_THREADS`, etc.): configuran el comportamiento desde fuera del programa.

La idea central es: el programador marca regiones del código que pueden ejecutarse en paralelo, y OpenMP se encarga de crear los hilos, repartir el trabajo y sincronizarlos.

---

## Requisitos y Compilación

- **GCC** con soporte OpenMP (viene incluido por defecto).
- Instalar en Ubuntu/Debian: `sudo apt-get install gcc`

**Compilación genérica:**

```bash
gcc -fopenmp -o output/ejercicioN ejercicioN.c
./output/ejercicioN
```

> El flag `-fopenmp` es **imprescindible**. Sin él, las directivas `#pragma omp` se ignoran silenciosamente y el programa se ejecuta de forma serial.

---

## Conceptos Clave de OpenMP

### Modelo Fork-Join

```
Hilo maestro (thread 0)
     |
     |--- #pragma omp parallel ──► Hilo 0   Hilo 1   Hilo 2   Hilo 3    ← FORK
     |                               |         |         |         |
     |                               | (trabajo en paralelo)       |
     |                               |         |         |         |
     |◄── barrera implícita ─────────┘─────────┘─────────┘─────────┘     ← JOIN
     |
  (continúa serial)
```

- **Fork**: al entrar a una región `parallel`, el hilo maestro crea un equipo de hilos.
- **Join**: al cerrar la región, todos los hilos se sincronizan (barrera implícita) y solo el hilo maestro continúa.
- El hilo maestro siempre tiene `thread_id = 0`.

### Serial vs Paralelo: cuándo conviene paralelizar

| Aspecto | Serial | Paralelo |
|---------|--------|----------|
| Hilos | 1 | N (configurables) |
| Overhead | Ninguno | Crear/destruir hilos, sincronización |
| Para N pequeño | ✅ Más rápido | ❌ El overhead domina |
| Para N grande | ❌ Lento | ✅ Se aprovecha el hardware |

**Regla práctica**: La paralelización solo conviene cuando el **trabajo útil** supera ampliamente al **overhead** de gestión de hilos. En los ejercicios del práctico, esto se observa a partir de ~10 millones de elementos.

**Speedup** = T_serial / T_paralelo. Un speedup de 2x significa que la versión paralela tarda la mitad.

### Directivas principales

| Directiva | Qué hace |
|-----------|----------|
| `#pragma omp parallel` | Crea una región paralela (fork). Todo el bloque se ejecuta por todos los hilos. |
| `#pragma omp for` | Dentro de una región paralela, reparte las iteraciones de un `for` entre los hilos. |
| `#pragma omp parallel for` | Combinación de `parallel` + `for` en una sola directiva. |
| `#pragma omp sections` | Reparte bloques de código independientes (`section`) entre hilos. |
| `#pragma omp single` | Solo un hilo ejecuta el bloque (para inicialización o I/O puntual). |
| `#pragma omp master` | Solo el hilo maestro (id=0) ejecuta el bloque; los demás lo ignoran. |
| `#pragma omp critical` | Sección crítica: exclusión mutua (un hilo a la vez). |
| `#pragma omp barrier` | Punto de sincronización: nadie pasa hasta que todos llegan. |
| `#pragma omp atomic` | Operación atómica sobre una variable (más ligero que critical). |

### Cláusulas de datos

| Cláusula | Comportamiento | Ejemplo de uso |
|----------|---------------|----------------|
| `shared(var)` | Todos los hilos ven la **misma** variable (misma dirección de memoria). Es el **default** para variables declaradas fuera de la región paralela. | Vectores que cada hilo lee/escribe en posiciones distintas. |
| `private(var)` | Cada hilo tiene su **propia copia**, **sin inicializar**. Al salir de la región, la variable original conserva su valor previo. | Contadores, IDs de hilo. |
| `firstprivate(var)` | Como `private`, pero la copia se **inicializa** con el valor que tenía antes de la región paralela. | Cuando necesitas que cada hilo arranque con un valor base. |
| `lastprivate(var)` | Como `private`, pero al salir la variable original toma el valor de la **última iteración**. | Guardar el resultado de la última iteración procesada. |
| `reduction(op:var)` | Cada hilo trabaja con una copia privada inicializada con el neutro de la operación. Al cerrar la región, OpenMP combina las copias con la operación indicada. | Sumas, productos, máximos, mínimos. |

### Sincronización: critical, barrier, atomic

Estas tres directivas resuelven problemas **distintos**. Es fundamental no confundirlas:

| Directiva | Problema que resuelve | Mecanismo | Costo |
|-----------|----------------------|-----------|-------|
| `critical` | Varios hilos **escriben la misma variable** a la vez → condición de carrera. | **Exclusión mutua**: solo un hilo a la vez ejecuta el bloque; los demás esperan turno. | Alto: serializa el acceso. |
| `barrier` | Un hilo **avanza a la fase siguiente** antes de que otros terminen la fase anterior. | **Sincronización temporal**: nadie cruza el punto hasta que **todos** los hilos del equipo llegan. | Medio: todos esperan al más lento. |
| `atomic` | Como `critical` pero para **una sola operación aritmética** sobre memoria. | Usa instrucciones atómicas del procesador (hardware). | Bajo: sin candado explícito. |

> **Regla mnemotécnica:**
> - `critical` ordena el **acceso** (uno a la vez, en cualquier orden).
> - `barrier` ordena el **tiempo** (todos juntos, en el mismo punto).

#### ¿Qué pasa si NO usas `critical` cuando es necesario?

Si varios hilos hacen `suma += 1` sin protección, la operación NO es atómica (son 3 pasos: leer, sumar, escribir). Dos hilos pueden leer el mismo valor viejo y una escritura pisa a la otra → **se pierden incrementos**. El resultado cambia en cada ejecución: esa es la firma de una **condición de carrera** (race condition).

#### ¿Qué pasa si NO usas `barrier` cuando es necesario?

Si un algoritmo tiene dos fases (ej: fase 1 = cada hilo escribe su dato, fase 2 = cada hilo lee los datos de todos), sin barrier el hilo más rápido llega a la fase 2 y lee datos que otros hilos todavía no escribieron → **resultados incorrectos e inconsistentes**.

#### ¿Por qué NO usar `critical` siempre?

Porque dentro del `critical` el programa **deja de ser paralelo**: los hilos hacen fila. Si todo el cuerpo del bucle está dentro de un `critical`, la versión "paralela" es más lenta que la secuencial. La sección crítica debe ser **lo más pequeña posible**.

### Reduction

La cláusula `reduction(op:variable)` es la forma correcta y eficiente de acumular resultados en paralelo.

**Funcionamiento interno:**
1. OpenMP crea una **copia privada** de la variable para cada hilo, inicializada con el **neutro** de la operación (`0` para `+`, `1` para `*`, `INT_MIN` para `max`, `INT_MAX` para `min`).
2. Cada hilo acumula en **su copia local** sin ningún tipo de sincronización (es privada).
3. Al terminar la región paralela, OpenMP **combina** todas las copias con la operación indicada en una **sola sincronización** final.

**Operaciones soportadas:** `+`, `-`, `*`, `&`, `|`, `^`, `&&`, `||`, `max`, `min`.

```c
// SIN reduction → CONDICIÓN DE CARRERA (resultado incorrecto)
int suma = 0;
#pragma omp parallel for
for (int i = 1; i <= 100; i++) {
    suma += i;  // múltiples hilos escriben la misma variable
}
// Resultado: impredecible (3664, 3833, 4586, 1778...)

// CON reduction → CORRECTO
int suma = 0;
#pragma omp parallel for reduction(+:suma)
for (int i = 1; i <= 100; i++) {
    suma += i;  // cada hilo suma en su copia privada
}
// Resultado: siempre 5050
```

### Schedule

Controla **cómo se reparten las iteraciones** de un `for` entre los hilos:

| Tipo | Reparto | Cuándo usarlo |
|------|---------|---------------|
| `static` | Las iteraciones se dividen **equitativamente al inicio** en bloques fijos. | Cuando todas las iteraciones cuestan lo mismo (ej: suma de vectores). Es el más eficiente por bajo overhead. |
| `dynamic` | Cada hilo **toma un bloque**, lo procesa y **pide el siguiente** que esté libre. | Cuando las iteraciones tienen **costo desigual** (ej: test de primalidad). |
| `guided` | Como `dynamic` pero los bloques empiezan grandes y van **decreciendo exponencialmente**. | Balance entre dynamic y static. Menos peticiones que dynamic al principio. |
| `auto` | El compilador/runtime decide. | Cuando no estás seguro. |

**Ejemplo con 4 hilos y N=1000:**
- `static`: Hilo 0 → [0,249], Hilo 1 → [250,499], Hilo 2 → [500,749], Hilo 3 → [750,999]
- `dynamic,100`: Hilo 0 pide [0,99], termina, pide [400,499]... el reparto depende de quién termina primero.

### Funciones de runtime

| Función | Qué devuelve |
|---------|-------------|
| `omp_get_thread_num()` | ID del hilo actual (0 a N-1). El 0 es el maestro. |
| `omp_get_num_threads()` | Total de hilos activos en la región paralela actual. **Fuera de una región paralela devuelve 1.** |
| `omp_get_max_threads()` | Máximo número de hilos que se pueden usar. |
| `omp_get_num_procs()` | Número de procesadores disponibles en el sistema. |
| `omp_set_num_threads(n)` | Establece la cantidad de hilos para la próxima región paralela. |
| `omp_get_wtime()` | Tiempo de reloj de pared en segundos (alta resolución). Se usa para medir rendimiento. |
| `omp_get_nested()` | Indica si el paralelismo anidado está habilitado (0 o 1). |

---

## Ejercicio 1 — Info del entorno OpenMP

> **Archivo:** `ejercicio1.c`
> **Compilar:** `gcc -fopenmp ejercicio1.c -o ejercicio1`

### ¿Qué problema resuelve?

Verificar que OpenMP está correctamente configurado y mostrar información del entorno: procesadores disponibles, hilos en uso, máximo de hilos y si el paralelismo anidado está disponible.

### ¿Cómo lo hace?

Crea una región paralela con `num_threads(2)` y dentro de ella consulta las funciones de runtime de OpenMP:

```c
#pragma omp parallel num_threads(cant_hilos)
{
    cant_procesadores = omp_get_num_procs();      // procesadores del sistema
    num_hilos_usando = omp_get_num_threads();      // hilos activos en esta región
    num_max_hilos = omp_get_max_threads();         // máximo permitido
    paralelismo_anidado = omp_get_nested();         // ¿anidado habilitado?
}
```

### Conceptos OpenMP que aplica

- `#pragma omp parallel num_threads(N)`: crea una región paralela con exactamente N hilos.
- Funciones de consulta del entorno: `omp_get_num_procs()`, `omp_get_num_threads()`, `omp_get_max_threads()`, `omp_get_nested()`.

### Observación importante

`omp_get_num_threads()` **devuelve el número real de hilos del equipo actual**, que puede ser distinto al máximo disponible si se lo configura con `num_threads()` o `omp_set_num_threads()`. Fuera de una región paralela, siempre devuelve 1.

---

## Ejercicio 2 — Cláusula private vs shared

> **Archivo:** `ejercicio2.c`
> **Compilar:** `gcc -fopenmp ejercicio2.c -o ejercicio2`

### ¿Qué problema resuelve?

Demostrar la diferencia entre variables compartidas (`shared`) y privadas (`private`) en OpenMP, y qué pasa cuando se omite la cláusula `private`.

### ¿Cómo lo hace?

Ejecuta dos bloques paralelos con 8 hilos, ambos asignando `valor = omp_get_thread_num()`:

**Caso 1 — SIN `private` (variable compartida por defecto):**

```c
#pragma omp parallel num_threads(8)
{
    valor = omp_get_thread_num();   // todos los hilos escriben la MISMA variable
    printf("Hilo %d ve su valor local = %d\n", omp_get_thread_num(), valor);
}
printf("Valor final fuera del bloque: %d\n", valor);
```

Todos los hilos escriben en la misma dirección de memoria. El valor final es **impredecible** (el del último hilo que escribió). Los `printf` pueden mostrar valores inconsistentes porque entre el `valor = ...` y el `printf` otro hilo puede haber sobrescrito la variable. **Esto es una condición de carrera**.

**Caso 2 — CON `private`:**

```c
#pragma omp parallel num_threads(8) private(valor)
{
    valor = omp_get_thread_num();   // cada hilo tiene SU propia copia
    printf("Hilo %d ve su valor local = %d\n", omp_get_thread_num(), valor);
}
printf("Valor final fuera del bloque: %d\n", valor);
// valor sigue siendo 10 (el valor original, no se modifica)
```

Cada hilo tiene una **copia independiente** (diferente dirección de memoria). No hay carrera. Al salir de la región paralela, la variable original **conserva su valor previo** (10).

### Resumen clave

| Aspecto | `shared` (default) | `private` |
|---------|-------------------|-----------|
| Memoria | Una sola copia, misma dirección para todos | Copia propia por hilo, distintas direcciones |
| Valor inicial dentro de la región | El valor actual de la variable | **Indefinido** (basura) |
| Valor después de la región | El de la última escritura (impredecible si hay carrera) | **Sin cambios** respecto a antes |
| Riesgo | Condición de carrera si se escribe sin protección | Ninguno (cada hilo trabaja aislado) |

---

## Ejercicio 3 — Schedule y conteo paralelo

> **Archivo:** `ejercicio3.c`
> **Compilar:** `gcc -fopenmp ejercicio3.c -o ejercicio3`

### ¿Qué problema resuelve?

Contar cuántos números son menores a 500 en un vector de N=100.000.000 elementos, comparando las tres políticas de schedule (`static`, `dynamic`, `guided`) para determinar cuál es más eficiente.

### ¿Cómo lo hace?

Inicializa un vector de 100M elementos con valores determinísticos entre 0 y 999. Luego ejecuta el conteo con cada política de schedule, usando `reduction(+:count)` para acumular de forma segura:

```c
// Ejemplo con schedule(static)
count = 0;
start = omp_get_wtime();
#pragma omp parallel for schedule(static) reduction(+:count)
for (int i = 0; i < N; i++) {
    if (vector[i] < 500) count++;
}
printf("Static : Menores a 500: %ld | Tiempo: %.6f s\n", count, omp_get_wtime() - start);
```

Se repite lo mismo con `schedule(dynamic, 10000)` y `schedule(guided, 10000)`.

### ¿Por qué `static` es la mejor opción para este problema?

**Porque la carga por iteración es totalmente homogénea**: evaluar `vector[i] < 500` cuesta exactamente lo mismo para cualquier `i`. No hay iteraciones "caras" y "baratas".

- **`static`**: reparto fijo al inicio, sin overhead de coordinación → **el más rápido**.
- **`dynamic`**: cada hilo pide bloques de a 10.000 → overhead de sincronización en cada solicitud, sin ningún beneficio porque el trabajo ya está balanceado.
- **`guided`**: bloques decrecientes → mismo overhead innecesario.

**`dynamic` y `guided` se justifican cuando las iteraciones tienen costo desigual**, por ejemplo: test de primalidad (probar si un número grande es primo tarda mucho más que uno pequeño), renderizado de fractales (algunos píxeles requieren muchas más iteraciones del escape test), o procesamiento de matrices esparcidas.

### Detalle: warm-up de caché

El código incluye un bucle de "warm-up" antes de medir:

```c
long dummy = 0;
#pragma omp parallel for schedule(static) reduction(+:dummy)
for (int i = 0; i < N; i++) dummy += vector[i];
```

Esto precarga el vector en la caché del procesador para que la primera medición (static) no pague el costo de traer los datos de RAM, lo cual sesgaría la comparación.

---

## Ejercicio 4 — Suma de vectores con reduction y comparación de hilos

> **Archivo:** `ejercicio4.c`
> **Compilar:** `gcc -fopenmp ejercicio4.c -o ejercicio4`

### ¿Qué problema resuelve?

Sumar dos vectores `C[i] = A[i] + B[i]` comparando la ejecución secuencial con la paralela usando 2, 4, 8 y 16 hilos, para diferentes tamaños de N (1M, 10M, 100M). Se calcula el **speedup** (aceleración) en cada caso.

### ¿Cómo lo hace?

```c
// Ejecución secuencial (línea base)
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}

// Ejecución paralela con X hilos
omp_set_num_threads(hilos);
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}

double speedup = tiempo_sec / tiempo_par;
```

### ¿Por qué NO necesita `critical` ni `reduction`?

Porque **cada hilo escribe en posiciones distintas** del vector C. La directiva `#pragma omp parallel for` garantiza que cada índice `i` se asigna a **un solo hilo**. No hay variable compartida que se modifique concurrentemente → no hay condición de carrera.

### Resultados y análisis (CPU con 4 hilos lógicos)

| N | Secuencial | 2 hilos | 4 hilos |
|---|-----------|---------|---------|
| 1.000.000 | 0.002s | 0.0006s (4.2x*) | 0.004s (0.6x) |
| 10.000.000 | 0.013s | 0.007s (1.8x) | 0.006s (2.1x) |
| 100.000.000 | 0.119s | 0.049s (2.4x) | — |

**Observaciones clave:**

1. **N = 1M con 4 hilos → peor que serial (0.6x)**: El overhead de crear y sincronizar hilos es mayor que el trabajo útil. La carga es demasiado liviana.
2. **N = 10M → punto óptimo**: El overhead se vuelve insignificante. El escalado con 2 hilos es casi lineal (1.8x).
3. **N = 100M → cuello de botella de memoria**: Se ve aceleración (2.4x con 2 hilos), pero no crece linealmente porque la operación es **memory-bound**: la RAM no puede alimentar operandos a los cores más rápido de lo que permite el bus de memoria.
4. **El speedup 4.2x con 2 hilos en N=1M** es una anomalía estadística, probablemente por alineación favorable en la caché L2/L3 o Turbo Boost asimétrico. No es reproducible consistentemente.

> **Concepto para la defensa:** Este ejercicio demuestra que la cláusula `reduction` NO es necesaria para suma de vectores (cada hilo toca posiciones distintas), pero SÍ es necesaria cuando todos los hilos acumulan sobre la misma variable escalar (como sumar todos los elementos, ejercicios 6 y 9).

---

## Ejercicio 5 — Sections: mínimo, máximo y promedio

> **Archivo:** `ejercicio5.c`
> **Compilar:** `gcc -fopenmp ejercicio5.c -o ejercicio5`

### ¿Qué problema resuelve?

Procesar un vector de N elementos (10.000, 100.000, 500.000) calculando su mínimo, máximo y promedio, usando **3 secciones paralelas** independientes para que cada tarea la haga un hilo distinto.

### ¿Cómo lo hace?

```c
omp_set_num_threads(3);  // 1 hilo por sección

#pragma omp parallel
{
    #pragma omp master
    {
        printf("Cantidad total de hilos en ejecucion = %d\n", omp_get_num_threads());
    }

    #pragma omp sections
    {
        #pragma omp section   // Sección 1: Máximo
        {
            double max_local = vec[0];
            for (int i = 1; i < N; i++)
                if (vec[i] > max_local) max_local = vec[i];
            max_val = max_local;
        }

        #pragma omp section   // Sección 2: Mínimo
        {
            double min_local = vec[0];
            for (int i = 1; i < N; i++)
                if (vec[i] < min_local) min_local = vec[i];
            min_val = min_local;
        }

        #pragma omp section   // Sección 3: Promedio (suma)
        {
            double suma_local = 0.0;
            for (int i = 0; i < N; i++)
                suma_local += vec[i];
            suma = suma_local;
        }
    }
}

promedio = suma / N;  // se calcula después de la región paralela
```

### Conceptos OpenMP que aplica

- **`#pragma omp sections`**: directiva de reparto de trabajo (*worksharing*) que asigna cada `section` a un hilo diferente.
- **`#pragma omp master`**: solo el hilo con id=0 ejecuta el bloque. A diferencia de `single`, **no tiene barrera implícita**, así que los otros hilos no esperan.
- **Barrera implícita** al cerrar `#pragma omp sections`: ningún hilo sigue hasta que las 3 secciones terminan. Esto garantiza que `min_val`, `max_val` y `suma` estén listos cuando se calcula el promedio fuera de la región paralela.

### ¿Qué pasa si hay más hilos que secciones?

Si usas 8 hilos en vez de 3, **solo 3 hilos trabajan** (uno por sección). Los otros 5 quedan ociosos esperando en la barrera implícita del cierre de `sections`. Se configura con 3 hilos porque hay exactamente 3 tareas independientes.

### ¿Por qué usa variables locales (`max_local`, `min_local`, `suma_local`)?

Para evitar **false sharing**: si los 3 hilos escribieran directamente `max_val`, `min_val` y `suma` (que son variables compartidas cercanas en memoria), cada escritura invalidaría la línea de caché de los otros cores, degradando el rendimiento. Usando variables locales en la pila, cada hilo trabaja en su propia caché y solo escribe el resultado final una vez.

### No determinismo en la salida

El orden de los `printf` de las secciones **cambia entre ejecuciones**. Ejemplo:

```
--> [Seccion 1 - Maximo] procesada por el hilo ID: 1     ← a veces sale antes
[Hilo Maestro ID 0]: Cantidad total de hilos = 3         ← a veces sale primero
--> [Seccion 3 - Promedio] procesada por el hilo ID: 0
--> [Seccion 2 - Minimo] procesada por el hilo ID: 2
```

Esto es normal: `#pragma omp master` **no frena a los otros hilos**. Mientras el hilo 0 hace su `printf`, los otros ya pueden estar ejecutando su sección.

---

## Ejercicio 6 — Máximo y mínimo con reduction(max/min)

> **Archivo:** `ejercicio6.c`
> **Compilar:** `gcc -fopenmp ejercicio6.c -o ejercicio6`

### ¿Qué problema resuelve?

Encontrar el máximo y el mínimo de un vector de N elementos (10.000, 100.000, 5.000.000) usando 4, 8 y 32 hilos, midiendo el tiempo en cada combinación.

### ¿Cómo lo hace?

Usa `reduction(max:max) reduction(min:min)` para que cada hilo encuentre el máximo y mínimo local de su porción, y OpenMP los combine al final:

```c
omp_set_num_threads(hilos);
int max = 0;
int min = INT_MAX;

#pragma omp parallel for reduction(max:max) reduction(min:min)
for (int i = 0; i < num; i++){
    if(v[i] > max) max = v[i];
    if(v[i] < min) min = v[i];
}
```

### ¿Qué pasaría SIN `reduction`?

Si `max` y `min` fueran simplemente `shared`, habría **condición de carrera**: dos hilos podrían leer el mismo `max`, ambos encontrar un valor mayor, y uno pisaría la escritura del otro. El resultado sería un máximo **menor** que el real (se perderían actualizaciones).

### ¿Por qué se prueban distintas cantidades de hilos?

Para observar cómo escala el rendimiento:
- **4 y 8 hilos**: se espera buen rendimiento si el sistema tiene suficientes cores.
- **32 hilos**: probablemente más hilos que cores físicos → los hilos compiten por CPU, se pierde eficiencia por cambios de contexto (*context switching*).

---

## Ejercicio 7 — Búsqueda paralela en vector desordenado

> **Archivo:** `ejercicio7.c`
> **Compilar:** `gcc -fopenmp ejercicio7.c -o ejercicio7`

### ¿Qué problema resuelve?

Buscar un valor específico en un vector desordenado de 400.000 y 1.000.000 elementos, usando 4, 8, 16 y 64 hilos, para encontrar la configuración que minimiza el tiempo de ejecución.

### ¿Cómo lo hace?

Usa un truco ingenioso con `reduction(min:encontrado)`:

```c
int encontrado = numV;  // valor centinela (= tamaño del vector, fuera de rango)

#pragma omp parallel for reduction(min:encontrado)
for (int i = 0; i < numV; i++){
    if (v[i] == num){
        encontrado = i;  // cada hilo que encuentra el valor guarda su índice
    }
}
// reduction(min:...) se queda con el MENOR índice encontrado
```

### ¿Por qué `reduction(min:...)` y no `critical`?

- Si hay **múltiples ocurrencias** del valor, varios hilos pueden encontrarlo en distintas posiciones. `reduction(min:encontrado)` devuelve la **primera posición** (la menor).
- Usar `critical` para proteger la escritura de `encontrado` sería correcto pero más lento, porque cada hilo que encuentre el valor tendría que esperar turno para escribir.
- Como solo necesitamos el índice mínimo, `reduction(min:...)` es la solución perfecta: cero sincronización durante el bucle, una sola combinación al final.

### Observación sobre la cantidad de hilos

Con más hilos que cores físicos (ej: 64 hilos en un CPU de 8 cores), el rendimiento puede **empeorar** por el overhead de scheduling y cambios de contexto. El punto óptimo suele estar cerca de la cantidad de cores físicos.

---

## Ejercicio 8 — critical y barrier

Este ejercicio tiene dos partes independientes que abordan los dos mecanismos de sincronización más importantes de OpenMP.

### 8A — critical: condición de carrera

> **Archivo:** `ejercicio8_critical.c`
> **Compilar:** `gcc -fopenmp -O0 ejercicio8_critical.c -o ejercicio8_critical`

#### ¿Qué problema resuelve?

Demostrar que una variable compartida modificada por varios hilos **sin protección** se corrompe (condición de carrera), y que `#pragma omp critical` lo soluciona.

#### ¿Cómo lo hace?

Ejecuta 1.000.000 de incrementos (`suma += 1`) en paralelo, primero sin protección y después con `critical`:

```c
omp_set_num_threads(4);

// CASO 1: SIN critical → INCORRECTO
#pragma omp parallel for private(i)
for (i = 0; i < N; i++) {
    suma_sin_critical += 1;   // ← CONDICIÓN DE CARRERA
}

// CASO 2: CON critical → CORRECTO
#pragma omp parallel for private(i)
for (i = 0; i < N; i++) {
    #pragma omp critical
    {
        suma_con_critical += 1;   // acceso serializado → seguro
    }
}
```

#### Salida real:

```
Valor esperado    : 1000000
Sin critical      : 812223  <-- SE PERDIERON INCREMENTOS
Con critical      : 1000000  (siempre correcto)
```

Se perdieron casi **190.000 incrementos**. Y el número **cambia en cada ejecución**: esa es la firma de una condición de carrera.

#### ¿Por qué ocurre?

`suma += 1` **parece una sola operación** pero el procesador la ejecuta en **3 pasos**:

```
1. LEER: registro = memoria[suma]     → registro = 500
2. SUMAR: registro = registro + 1      → registro = 501
3. ESCRIBIR: memoria[suma] = registro  → memoria = 501
```

Si dos hilos leen `500` al mismo tiempo, ambos calculan `501`, ambos escriben `501`. Se hicieron 2 incrementos y el contador avanzó solo 1. **Se perdió una suma**.

#### ¿Por qué se compila con `-O0`?

Con optimización agresiva (`-O2`, `-O3`), el compilador puede mantener `suma` en un registro y cambiar el patrón del error, haciéndolo menos visible o inconsistente. Con `-O0` la carrera se manifiesta de forma consistente y clara.

#### ¿Por qué NO usar `critical` siempre?

Porque dentro del `critical` el programa **deja de ser paralelo**: los hilos hacen fila. En este ejemplo, **todo** el cuerpo del bucle está protegido, así que la versión con `critical` es **más lenta que la secuencial**.

**Alternativas más eficientes:**
- **`reduction(+:suma)`**: cada hilo acumula en una copia privada, una sola sincronización al final. **La mejor opción para acumuladores**.
- **`#pragma omp atomic`**: para una sola operación aritmética, usa instrucciones atómicas del hardware. Más ligero que `critical`.

| Mecanismo | Correctitud | Rendimiento | Cuándo usarlo |
|-----------|------------|-------------|---------------|
| Sin protección | ❌ | ✅ (rápido pero mal) | Nunca (si hay escritura compartida) |
| `critical` | ✅ | ❌ (serializa todo) | Bloques de varias líneas que deben ejecutarse juntas |
| `atomic` | ✅ | ⚠️ (razonable) | Una sola operación aritmética simple |
| `reduction` | ✅ | ✅ (lo mejor) | Acumulaciones (+, *, max, min) en bucles |

---

### 8B — barrier: sincronización entre fases

> **Archivo:** `ejercicio8_barrier.c`
> **Compilar:** `gcc -fopenmp ejercicio8_barrier.c -o ejercicio8_barrier`

#### ¿Qué problema resuelve?

Demostrar que cuando un algoritmo tiene **dos fases dependientes** (primero escribir, luego leer), sin `barrier` los hilos rápidos leen datos que todavía no fueron escritos por los lentos.

#### ¿Cómo lo hace?

Crea 4 hilos, cada uno escribe su casilla en un vector compartido (FASE 1) y luego suma todo el vector (FASE 2). Un `usleep` hace que los hilos terminen la FASE 1 en momentos distintos:

```c
#define NH 4
int v[NH];    // vector compartido

#pragma omp parallel num_threads(NH)
{
    int tid = omp_get_thread_num();
    
    // FASE 1: cada hilo escribe SOLO su casilla
    usleep(tid * 100000);    // hilo 0 es rápido, hilo 3 es lento
    v[tid] = tid + 1;
    
    // FASE 2: cada hilo lee TODO el vector
    // SIN barrier → el hilo 0 lee antes de que los otros escriban
    // CON barrier → todos esperan, luego leen el vector completo
    
    #pragma omp barrier    // ← SOLO en el caso 2
    
    int suma = 0;
    for (int j = 0; j < NH; j++) suma += v[j];
    printf("Hilo %d -> suma leida = %d (esperado 10)\n", tid, suma);
}
```

#### Salida real:

```
--- SIN barrier ---
Hilo 0 -> suma leida = -2 (esperado 10)     ← leyó 1 + (-1) + (-1) + (-1)
Hilo 1 -> suma leida = 1  (esperado 10)     ← leyó 1 + 2 + (-1) + (-1)
Hilo 2 -> suma leida = 5  (esperado 10)     ← leyó 1 + 2 + 3 + (-1)
Hilo 3 -> suma leida = 10 (esperado 10)     ← el último ve todo completo

--- CON barrier ---
Hilo 3 -> suma leida = 10 (esperado 10)     ← todos ven 10
Hilo 1 -> suma leida = 10 (esperado 10)
Hilo 2 -> suma leida = 10 (esperado 10)
Hilo 0 -> suma leida = 10 (esperado 10)
```

#### ¿Por qué `-2` en el hilo 0?

El vector se inicializó con `v[i] = -1` ("sin escribir"). El hilo 0 es el más rápido (no hace `usleep`), escribe `v[0] = 1` e inmediatamente lee todo el vector. Los otros hilos todavía no escribieron, así que lee `1 + (-1) + (-1) + (-1) = -2`.

#### El `usleep` no es parte del algoritmo

Solo exagera la desincronización para que el error sea **visible y reproducible**. Sin él, el error igualmente existe pero aparecería esporádicamente.

#### Errores clásicos con barrier:

1. **Deadlock**: si el `barrier` queda dentro de un `if (tid == 0)` o un `single`, los hilos que no entran al bloque **nunca llegan al barrier**, y los que sí llegan **esperan para siempre**.
2. **Barrier redundante**: `for`, `sections`, `single` y el cierre de la región `parallel` ya tienen un barrier implícito. Agregar uno explícito ahí no hace nada. Se elimina con la cláusula `nowait`.

---

## Ejercicio 9 — Análisis de código: parallel + for + schedule(dynamic)

> **Archivo:** `ejercicio9.c`
> **Compilar:** `gcc -fopenmp ejercicio9.c -o ejercicio9`

### ¿Qué problema resuelve?

Este ejercicio consiste en **analizar y explicar** un código dado (no escribirlo). Es una suma de vectores `c[i] = a[i] + b[i]` con N=100, diseñada para estudiar la estructura de una región paralela con reparto dinámico.

### El código y su análisis cláusula por cláusula

```c
#define CHUNKSIZE 10
#define N 100

int main(int argc, char *argv[])
{
    int nthreads, tid, i, chunk;
    float a[N], b[N], c[N];

    // Inicialización SECUENCIAL (fuera de la región paralela)
    for (i = 0; i < N; i++)
        a[i] = b[i] = i * 1.0;

    chunk = CHUNKSIZE;

    // FORK: se crea el equipo de hilos
    #pragma omp parallel shared(a, b, c, nthreads, chunk) private(i, tid)
    {
        tid = omp_get_thread_num();
        
        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }

        printf("Thread %d starting...\n", tid);

        // WORK-SHARING: las N iteraciones se REPARTEN, no se repiten
        #pragma omp for schedule(dynamic, chunk)
        for (i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
            printf("Thread %d: c[%d]= %f\n", tid, i, c[i]);
        }

    }   // JOIN: barrera implícita, los hilos se sincronizan y mueren
}
```

### Desglose de cada elemento

| Elemento | Qué hace y por qué |
|----------|-------------------|
| `#pragma omp parallel` | Marca el **fork**: crea el equipo de hilos. Todo lo que está adentro lo ejecuta **cada hilo completo**, salvo lo repartido por `omp for`. |
| `shared(a, b, c)` | Los vectores son una única copia en memoria. Cada hilo lee/escribe posiciones distintas → no hay carrera. |
| `shared(nthreads)` | La escribe el hilo 0 para reportar el tamaño del equipo. |
| `shared(chunk)` | Parámetro de configuración: todos leen el mismo valor (10). Solo lectura. |
| `private(tid)` | **Obligatorio**: cada hilo tiene su propio id. Si fuera shared, se pisarían. |
| `private(i)` | Cada hilo recorre su propio rango de iteraciones. Nota: la variable de control de un `omp for` es privada automáticamente, pero acá está explícita porque `i` fue declarada fuera. |
| `#pragma omp for` | **Reparte** las 100 iteraciones entre los hilos. **Sin** esta directiva, cada hilo ejecutaría las 100 → se harían 400 sumas repetidas. |
| `schedule(dynamic, 10)` | Cada hilo toma un bloque de 10 iteraciones. Al terminarlo, pide el siguiente libre. Reparto en tiempo de ejecución. |

### Diferencia clave: `parallel` vs `parallel for`

- **`#pragma omp parallel`**: crea el equipo. Todos ejecutan el mismo bloque completo.
- **`#pragma omp for`**: dentro de una región paralela existente, reparte iteraciones entre los hilos.
- **`#pragma omp parallel for`**: combinación de ambas en una sola directiva.

En el ejercicio 9 están **separadas** porque hay código previo al `for` (los `printf` de "starting...") que todos los hilos deben ejecutar. Si se usara `parallel for`, solo se paralelizaría el bucle y se perdería la región donde cada hilo se presenta.

### Puntos de sincronización (sin escribir barrier explícito)

1. **Barrier implícito al final del `omp for`**: ningún hilo pasa del bucle hasta que todos terminaron sus iteraciones.
2. **Barrier implícito al cerrar `}` de `parallel`** (el join): todos los hilos se sincronizan y mueren.

El primero se podría eliminar con `nowait`; el del cierre de la región no.

### ¿Por qué `dynamic` es una mala elección para este problema?

Todas las iteraciones cuestan exactamente lo mismo (una suma). `dynamic` solo agrega el overhead de pedir bloques sin ningún beneficio. `static` sería más rápido.

`dynamic` se justifica cuando el **costo por iteración es desigual**: si iterar sobre `i=7` tarda 10 veces más que `i=3`, un reparto estático dejaría hilos ociosos mientras otros trabajan. Con `dynamic`, el hilo que termina rápido toma más trabajo.

### Salida y cómo interpretarla

```
Thread 2 starting...
Thread 2: c[0]= 0.000000
Thread 2: c[1]= 2.000000
...
Number of threads = 4
Thread 0 starting...
Thread 0: c[20]= 40.000000
```

Tres cosas para la defensa:

1. **La salida es desordenada y cambia en cada ejecución**: los `printf` de distintos hilos se intercalan. Eso no es un error.
2. **Los índices vienen de a bloques de 10**: 0–9, 10–19, 20–29... esa es la huella del `chunk = 10`.
3. **Las 100 iteraciones se hacen UNA sola vez en total**, no 100 por hilo. Se comprueba contando las líneas `c[...]`: son exactamente 100.

---

## example.c — Ejemplo mínimo de reduction

> **Archivo:** `example.c`
> **Compilar:** `gcc -fopenmp example.c -o example`

Ejemplo mínimo que demuestra `reduction(+:suma)`: suma los números del 1 al 100 en paralelo.

```c
int suma = 0;
#pragma omp parallel for reduction(+:suma)
for (int i = 1; i <= 100; i++){
    suma += i;
}
printf("Suma = %d\n", suma);   // Siempre 5050
```

Sin `reduction`, el resultado sería impredecible (condición de carrera). Con `reduction`, cada hilo acumula en su copia privada y OpenMP combina al final → resultado siempre correcto.

---

## Resumen para la defensa

### Tabla resumen de directivas y cuándo usarlas

| Necesidad | Directiva/Cláusula | Ejemplo en el práctico |
|-----------|-------------------|----------------------|
| Paralelizar un bucle | `#pragma omp parallel for` | Ej. 3, 4, 6, 7 |
| Acumular resultados (suma, max, min) | `reduction(op:var)` | Ej. 3, 6, 7, 9 (example.c) |
| Proteger escritura compartida | `#pragma omp critical` | Ej. 8A |
| Sincronizar entre fases | `#pragma omp barrier` | Ej. 8B |
| Repartir tareas distintas | `#pragma omp sections` | Ej. 5 |
| Controlar reparto de iteraciones | `schedule(static/dynamic/guided)` | Ej. 3, 4, 9 |
| Que cada hilo tenga su variable | `private(var)` | Ej. 2, 9 |

### Preguntas frecuentes de defensa

**¿Por qué en la suma de vectores no necesito `critical`?**
Porque cada hilo escribe en posiciones **distintas** de C (`C[i]` con diferentes `i`). `critical` es para cuando múltiples hilos escriben **la misma variable**.

**¿Cuál es la diferencia entre `parallel` y `parallel for`?**
`parallel` crea hilos y todos ejecutan el mismo bloque. `for` reparte iteraciones. `parallel for` combina ambos.

**¿Dónde hay sincronización si no escribí `barrier`?**
Al final de `omp for` y al cerrar `omp parallel` hay barriers **implícitos**.

**¿Cómo demuestro que hay condición de carrera?**
Corriendo el mismo binario varias veces: el resultado incorrecto **cambia**. Con 1 hilo (`OMP_NUM_THREADS=1`) siempre da bien. Un error de lógica daría siempre el mismo resultado equivocado.

**¿`critical` o `atomic`?**
`atomic` es más rápido (usa hardware) pero solo sirve para **una operación aritmética simple**. `critical` protege **bloques arbitrarios** de código con un candado.

**¿Por qué `critical` puede ser peor que serial?**
Porque serializa el acceso: los hilos hacen fila. Si el bloque protegido es el **todo** cuerpo del bucle, el programa paralelo es más lento que el secuencial por el overhead del candado.

**¿Qué pasa si pongo `barrier` dentro de un `if`?**
**Deadlock**: los hilos que no entran al `if` nunca llegan al barrier, y los que sí llegan esperan para siempre.

**¿`static` o `dynamic`?**
`static` para cargas homogéneas (mismo costo por iteración). `dynamic` para cargas heterogéneas (iteraciones con costo variable). `static` tiene menos overhead.

---

*Compilación general:* `gcc -fopenmp -o ejercicioN ejercicioN.c`
*Cantidad de hilos:* se controla con `omp_set_num_threads()`, la cláusula `num_threads()`, o la variable de entorno `OMP_NUM_THREADS`.