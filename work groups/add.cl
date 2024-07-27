__kernel
void vec_add(
         int    n,
__global float* a,
__global float* b,
__global float* c){

    int i = get_global_id(0);

    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}