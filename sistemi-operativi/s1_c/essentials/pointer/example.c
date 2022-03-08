#include <stdio.h>
#include <stdlib.h>

int func (int arr[]) {
    return 0;
}

int main () {
    // Pointers
        int m = 3;
        int *p = &m;    /* Store the address of m var in p */
        int n = *p;

        printf ("m  = 3   -->  m  = %d\n", m);
        printf ("*p = &m  -->  p* = %d\n", *p);
        printf ("n  = *p  -->  n  = %d\n", n);
        // n=&m;                /* This will cause an error! */
        printf ("&m = %ls\n", &m);

    // Pointer and array
        int arr[10], *p1, *p2;
        p1 = &arr[2];
        p2 = p1 + 3;    /* now p2 points to arr[5] */
        p1 = p2 - 1;    /* now p2 points to arr[4] */
        *p1 = 40;       /* arr[4] = 40 */
        *p2 = *p1;      /* arr[5] = arr[4] */

        printf ("&arr[4] = %ls\n", &arr[4]);
        printf ("\n");
    
    // Pointers and first array element
        func (arr); // This line is equal to: func (&arr[0])
        // arr is a pointer to te first element

    // Others about pointers
        char *str = malloc(5);
        str[0] = 'y';   /* value: y  - address 0x23 */
        str[1] = 'e';   /* value: e  - address 0x24 */
        str[2] = 'a';   /* value: a  - address 0x25 */
        str[3] = 'h';   /* value: h  - address 0x26 */
        str[4] = '\0';  /* value: \0 - address 0x27 */
        free(str);      /* All done now */

        int tmp = 420;                  /* tmp: value = 420 - address: 0xaddres1 */
        int *tmpPointer = &val;         /* tmpPointer: value = 0xaddres1 - address: 0xaddres2 */
        if (&tmp == tmpPointer)
            printf ("tmpPointer = %ls\n", tmpPointer);
        printf ("\n");


    return 0;
}
