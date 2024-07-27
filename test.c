#include "clinit.h"

int main(){
    CL cl;
    cl.init("test.cl", { "vec_add"});
    return 0;
}