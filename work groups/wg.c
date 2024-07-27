#include "clinit.h"

#define COUNT (1 << 17)
#define RUNS  1000

float rand_float(){
    return (float)rand() / RAND_MAX;
}

void rand_float_arr(float *arr, size_t count){
    for(size_t i = 0; i < count; i++)
        arr[i] = rand_float();
}

int main(){
    CL cl;
    cl.init("add.cl", { "vec_add" });

    float veca[COUNT];
    float vecb[COUNT];
    float vecc[COUNT];

    rand_float_arr(veca, COUNT);
    rand_float_arr(vecb, COUNT);

    cl_mem veca_clmem = cl.alloc_buffer(
        "veca", sizeof(veca), veca, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

    cl_mem vecb_clmem = cl.alloc_buffer(
        "vecb", sizeof(vecb), vecb, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

    cl_mem vecc_clmem = cl.alloc_buffer(
        "vecc", sizeof(vecc), NULL, CL_MEM_WRITE_ONLY);
        
    int n = 100000;
    clSetKernelArg(cl.kernels[0], 0, sizeof(int), &n);
    clSetKernelArg(cl.kernels[0], 1, sizeof(cl_mem), &veca_clmem);
    clSetKernelArg(cl.kernels[0], 2, sizeof(cl_mem), &vecb_clmem);
    clSetKernelArg(cl.kernels[0], 3, sizeof(cl_mem), &vecc_clmem);

    size_t global_work[] = { COUNT };
    size_t local_work[]  = { cl.max_workgroup_size };

    double naive_time = 0.0; 
    double opt_time   = 0.0; 
    
    for (int i = 0; i < RUNS; i++){
        naive_time += cl.time_execution(0, 1, global_work, NULL);
    }

    // for (int i = 0; i < RUNS; i++){
    //     opt_time += cl.time_execution(0, 1, global_work, local_work);
    // }

    naive_time /= RUNS;
    opt_time   /= RUNS;

    std::cout << "Single work group:\t"    << naive_time << std::endl;
    std::cout << "Multiple work groups:\t" << opt_time   << std::endl;

    cl.free();
    clReleaseMemObject(veca_clmem);
    clReleaseMemObject(vecb_clmem);
    clReleaseMemObject(vecc_clmem);

    return 0;
}