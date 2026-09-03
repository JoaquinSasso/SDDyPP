#include <stdio.h>
#include <omp.h>

int main()
{
   int suma = 0;
   #pragma omp parallel for reduction (+:suma)
   for (int i = 1; i <= 100; i++){
      suma += i;
   }
   printf("Suma = %d\n", suma);
   return 0;
}

//gcc -fopenmp -o example.exe example.c && ./example.exe