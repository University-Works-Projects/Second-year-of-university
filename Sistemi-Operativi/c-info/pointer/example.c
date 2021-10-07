#include <stdio.h>

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
        printf ("&m = %d\n", &m);

    // Pointer and array
        int arr[10], *p1, *p2;
        p1 = &arr[2];
        p2 = p1 + 3;    /* now p2 points to arr[5] */
        p1 = p2 - 1;    /* now p2 points to arr[4] */
        *p1 = 40;       /* arr[4] = 40 */
        *p2 = *p1;      /* arr[5] = arr[4] */

        printf ("&arr[4] = %d\n", &arr[4]);
    
    // Pointers and first array element
        func (arr); // This line is equal to: func (&arr[0])
        // arr is a pointer to te first element

    return 0;
}
