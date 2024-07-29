#include <clinit.h>

#include <string>

#define COUNT     100
#define BATCHSIZE 5

#define WIDTH    128
#define HEIGHT   128
#define CHANNELS 3
#define FEATURES 64

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
    char options[128];
    // std::snprintf(options, 128, "-DFILTERW=%d -DFILTERH=%d -DCHANNELS=%d", 
    //     FILTERWIDTH, FILTERHEIGHT, CHANNELS);

    std::snprintf(options, 128, "-DFILTERW=%d -DFILTERH=%d -DCHANNELS=%d -DBSIZE=%d -DPREVW=%d -DPREVH=%d -DSTRIDEX=%d -DSTRIDEY=%d",
        FILTERWIDTH, FILTERHEIGHT, CHANNELS, BATCHSIZE, WIDTH, HEIGHT, STRIDEX, STRIDEY);

    CL cl("conv.cl", options, { "convolution", "convolution_opt" });

    float *image      = new float[IMAGESIZE];
    float *org_output = new float[OUTSIZE];
    float *opt_output = new float[OUTSIZE];
    float *filters    = new float[FILTERSIZE];

    rand_array(image,    IMAGESIZE);
    rand_array(filters,  FILTERSIZE);

    cl_mem image_clmem = cl.alloc_buffer(
        "image", IMAGESIZE * sizeof(float), image, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

    cl_mem org_output_clmem = cl.alloc_buffer(
        "org_output", OUTSIZE * sizeof(float), NULL, CL_MEM_WRITE_ONLY);

    cl_mem opt_output_clmem = cl.alloc_buffer(
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

    std::cout << "Original time:\t" << original_time << "ms" << std::endl;

    // Optimised kernel args
    cl.set_arg_clmem(1, 0, filters_clmem);
    cl.set_arg_clmem(1, 1, image_clmem);
    cl.set_arg_clmem(1, 2, opt_output_clmem);

    size_t optimised_global_size[] = { BATCHSIZE * OUTY, OUTX, FEATURES };
    size_t optimised_local_size[]  = { 1, 1, FEATURES };

    double optimised_time = 0.0;
    for (int i = 0; i < COUNT; i++){
        optimised_time += cl.time_execution(1, 3,
            optimised_global_size, NULL);
    }

    std::cout << "Optimised time:\t" << optimised_time << "ms" << std::endl;

    // Verify the results
    clEnqueueReadBuffer(cl.command_queue, org_output_clmem, CL_TRUE, 0,
        OUTSIZE * sizeof(float), org_output, 0, NULL, NULL);
    
    clEnqueueReadBuffer(cl.command_queue, opt_output_clmem, CL_TRUE, 0,
        OUTSIZE * sizeof(float), opt_output, 0, NULL, NULL);
    
    clFinish(cl.command_queue);

    for (int i = 0; i < OUTSIZE; i++){
        if (org_output[i] != opt_output[i]){
            std::cout << "Mismatch at index " << i << std::endl;
            break;
        }
    }

    clReleaseMemObject(image_clmem);
    clReleaseMemObject(org_output_clmem);
    clReleaseMemObject(opt_output_clmem);
    clReleaseMemObject(filters_clmem);

    delete[] image;
    delete[] org_output;
    delete[] opt_output;    
    delete[] filters;
}