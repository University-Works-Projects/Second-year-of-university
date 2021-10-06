#include <stdio.h>

int k () {
    auto int t = 3;
    return t;
}

int main () {

    printf("%d",k());
    printf ("\n");

    return 0;
}