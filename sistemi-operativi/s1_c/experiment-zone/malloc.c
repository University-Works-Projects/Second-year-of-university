#include <stdio.h>
#include <stdlib.h>

int main () {
    int *p, m;
    p = (int *) malloc(sizeof(int));    /* Allocate 4 bytes */
    printf("Enter a number: ");
    scanf("%d", p);
    printf("*p = %d\n", *p);
    free(p);

    return 0;
}