#include <clinit.h>

#define COUNT     10
#define BATCHSIZE 5

#define WIDTH    1024
#define HEIGHT   1024
#define CHANNELS 3
#define FEATURES 4

#define FILTERWIDTH  5
#define FILTERHEIGHT 5

#define STRIDEX 1
#define STRIDEY 1

#define masks(prev_size, mask_size, stride) \
    ((prev_size - mask_size) / stride + 1) 

#define IMAGESIZE   BATCHSIZE * WIDTH * HEIGHT * CHANNELS
#define FILTERSIZE  FILTERWIDTH * FILTERHEIGHT * CHANNELS * FEATURES

#define OUTX masks(WIDTH, FILTERWIDTH, STRIDEX)
#define OUTY masks(HEIGHT, FILTERHEIGHT, STRIDEY)

#define OUTSIZE BATCHSIZE * OUTY * OUTX * FEATURES

void rand_array(float *arr, int size){
    for(int i = 0; i < size; i++){
        arr[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main(){
    CL cl("conv.cl", {"convolution"});

    float *image      = new float[IMAGESIZE];
    float *org_output = new float[OUTSIZE];
    float *filters    = new float[FILTERSIZE];

    rand_array(image,    IMAGESIZE);
    rand_array(filters,  FILTERSIZE);

    cl_mem image_clmem = cl.alloc_buffer(
        "image", IMAGESIZE * sizeof(float), image, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

    cl_mem org_output_clmem = cl.alloc_buffer(
        "org_output", OUTSIZE * sizeof(float), NULL, CL_MEM_WRITE_ONLY);
    
    cl_mem filters_clmem = cl.alloc_buffer(
        "filters", FILTERSIZE * sizeof(float), filters, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

    // Original, unoptimised kernel args
    cl.set_arg_int(0, 0, WIDTH);
    cl.set_arg_int(0, 1, HEIGHT);
    cl.set_arg_int(0, 2, masks(HEIGHT, FILTERHEIGHT, STRIDEY));
    cl.set_arg_int(0, 3, FILTERWIDTH);
    cl.set_arg_int(0, 4, FILTERHEIGHT);
    cl.set_arg_int(0, 5, CHANNELS);
    cl.set_arg_int(0, 6, STRIDEX);
    cl.set_arg_int(0, 7, STRIDEY);
    cl.set_arg_clmem(0, 8, filters_clmem);
    cl.set_arg_clmem(0, 9, image_clmem);
    cl.set_arg_clmem(0, 10, org_output_clmem);

    size_t original_global_size[] = { BATCHSIZE * OUTY, OUTX, FEATURES };

    double original_time = 0.0;
    for (int i = 0; i < COUNT; i++){
        original_time += cl.time_execution(0, 3,
            original_global_size, NULL);
    }

    std::cout << "Original time: " << original_time << "ms" << std::endl;

    clReleaseMemObject(image_clmem);
    clReleaseMemObject(org_output_clmem);
    clReleaseMemObject(filters_clmem);

    delete[] image;
    delete[] org_output;    
    delete[] filters;
}