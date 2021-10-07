#include <stdio.h>
#include <stdlib.h>

void print_prime (int m) {
    int i, j;
    int stop;
    char *ary = malloc(m);
    
    if (ary = NULL) return -1;
    for (i = 0; i < m; i++)
        ary[i] = 1;
    ary[0] = ary[1] = 0;
    ary[2];
    stop = 0;
    for (i = 3; i < m; i++) {
        for (j = 2; j < i && !stop; j++) {
            if (ary[j] && i % j == 0) {
                ary[j] = 0;
                stop = 1;
            }
        }  
    }

    for (i = 0; i < m; i++)
        if (ary[i])
            printf("%d",i);
    free(ary);
}
