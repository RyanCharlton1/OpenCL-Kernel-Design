#include <clinit.h>

#include <string>

#define COUNT     100

#define CHANNELS 64
#define FEATURES 128

#define FILTERWIDTH  128
#define FILTERHEIGHT 128

#define FILTERSIZE  FILTERWIDTH * FILTERHEIGHT * CHANNELS * FEATURES

void rand_array(float *arr, int size){
    for(int i = 0; i < size; i++){
        arr[i] = (float)rand() / (float)RAND_MAX;
    }
}

std::string print_filters(float* filters){
    char buffer[16];
    std::string s;

    for (int f = 0; f < FEATURES; f++){
        int filter_index = f * FILTERWIDTH * FILTERHEIGHT * CHANNELS;

        for (int y = 0; y < FILTERHEIGHT; y++){
            int height_index = filter_index + y * FILTERWIDTH * CHANNELS;
            s += "[";

            for (int x = 0; x < FILTERWIDTH; x++){
                int width_index = height_index + x * CHANNELS;
                s += "[";

                for (int c = 0; c < CHANNELS; c++){
                    int channel_index = width_index + c;

                    std::snprintf(buffer, 16, "% .2f, ", filters[channel_index]);
                    s += buffer;
                }
                s.pop_back();
                s.pop_back();

                s += "], ";
            }

            s.pop_back();
            s.pop_back();
            s += "]\n";
        }

        s += "\n";
    }

    return s;
}

int main(){
    char options[128];
    std::snprintf(options, 128, "-DFILTERW=%d -DFILTERH=%d -DCHANNELS=%d",
        FILTERWIDTH, FILTERHEIGHT, CHANNELS);

    CL cl("rev.cl", options, { "rev_elem", "rev_channel" });

    float *filters     = new float[FILTERSIZE];
    float *rev_elem    = new float[FILTERSIZE];
    float *rev_channel = new float[FILTERSIZE];

    rand_array(filters, FILTERSIZE);

    cl_mem filters_clmem = cl.alloc_buffer(
        "filters", FILTERSIZE * sizeof(float), filters, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

    cl_mem rev_elem_clmem = cl.alloc_buffer(
        "rev_elem", FILTERSIZE * sizeof(float), NULL, CL_MEM_WRITE_ONLY);

    cl_mem rev_channel_clmem = cl.alloc_buffer(
        "rev_channel", FILTERSIZE * sizeof(float), NULL, CL_MEM_WRITE_ONLY);

    size_t global_size[] = { FEATURES * FILTERHEIGHT, FILTERWIDTH, CHANNELS };

    // Elementwise reverse args
    cl.set_arg_clmem(0, 0, filters_clmem);
    cl.set_arg_clmem(0, 1, rev_elem_clmem);

    double elem_time = 0.0;
    for (int i = 0; i < COUNT; i++)
        elem_time += cl.time_execution(0, 3, global_size, NULL);
    
    std::cout << "Elementwise reverse time:\t" << elem_time << "ms" << std::endl;

    // Channelwise reverse args
    cl.set_arg_clmem(1, 0, filters_clmem);
    cl.set_arg_clmem(1, 1, rev_channel_clmem);

    double channel_time = 0.0;
    for (int i = 0; i < COUNT; i++)
        channel_time += cl.time_execution(1, 3, global_size, NULL);

    std::cout << "Channelwise reverse time:\t" << channel_time << "ms" << std::endl;

    // Verify results

    clEnqueueReadBuffer(cl.command_queue, rev_elem_clmem, 
        CL_TRUE, 0, FILTERSIZE * sizeof(float), rev_elem, 0, NULL, NULL);

    clEnqueueReadBuffer(cl.command_queue, rev_channel_clmem,
        CL_TRUE, 0, FILTERSIZE * sizeof(float), rev_channel, 0, NULL, NULL);

    clFinish(cl.command_queue);

    // Verify correctness manually 
    // std::cout << print_filters(filters) << std::endl;
    // std::cout << print_filters(rev_elem) << std::endl;

    // Check both methods produce the same result
    for (int i = 0; i < FILTERSIZE; i++){
        if (rev_elem[i] != rev_channel[i]){
            std::cout << "Mismatch at index " << i << std::endl;
            break;
        }
    }

    clReleaseMemObject(filters_clmem);
    clReleaseMemObject(rev_elem_clmem);
    clReleaseMemObject(rev_channel_clmem);

    delete[] filters; 
    delete[] rev_elem;
    delete[] rev_channel;
}