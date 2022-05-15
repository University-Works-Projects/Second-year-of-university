#include <iostream>

using namespace std;

/*  2006_05_25_es_5
class A {
protected:
public:
    int x;
    A () { this->x = 10; }
    void f() {
        cout << this->x << endl;
    }
    void g() { f(); }
};

class B : public A {
public:
    int x = 20;
    B () { this->x = 20; }
    void f() {
        cout << this->x << endl;
    }
};
*/

int x = 3;
void temp (int y) {
    x=x-1;
    y=y+10;
    x=x+y;
    cout << "first: " << x << endl;
}

int main () {

    int v[10];
    int x = 4;
    for (int i = 0; i < 10; i++) { v[i] = i; }
    v[6] = v[5] + v[4];
    for (int i = 0; i < 10; i++) { cout << v[i] << " "; }
    cout << endl << x << endl <<  v[6] << endl;

    /* 2006_05_25_es_5
    A a;
    a.g();
    a.f();
    B b;
    cout << "BEFORE:: a.x = " << a.x << "; b.x = " << b.x;
    a = b;
    cout << endl << "AFTER::  a.x = " << a.x << "; b.x = " << b.x << endl;
    a.g();
    a.f();
    printf("%d%c", a.x, '\n');
    printf("%d%c", b.x, '\n');
    b.g();
    */

    return 0;
}