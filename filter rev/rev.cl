__kernel
void rev_elem(
__global float* filters,
__global float* reversed){
    
    int filtery = get_global_id(0);
    int filterx = get_global_id(1);
    int filterc = get_global_id(2);

    int feature = filtery / FILTERH;

    int input_index = filtery * FILTERW * CHANNELS + filterx * CHANNELS + filterc;

    filtery %= FILTERH;

    int output_index;
    output_index  = feature * FILTERH * FILTERW * CHANNELS;
    output_index += (FILTERH - 1 - filtery) * FILTERW * CHANNELS;
    output_index += (FILTERW - 1 - filterx) * CHANNELS;
    output_index += filterc;

    reversed[input_index] = filters[output_index];
}

__kernel 
void rev_channel(
__global float* filters,
__global float* reversed){

    int filtery = get_global_id(0);
    int filterx = get_global_id(1);

    int feature = filtery / FILTERH;

    int input_index = filtery * FILTERW * CHANNELS + filterx * CHANNELS;
    
    filtery %= FILTERH;

    int output_index;
    output_index  = feature * FILTERH * FILTERW * CHANNELS;
    output_index += (FILTERH - 1 - filtery) * FILTERW * CHANNELS;
    output_index += (FILTERW - 1 - filterx) * CHANNELS;

    #pragma unroll
    for (int i = 0; i < CHANNELS; i++){
        reversed[input_index] = filters[output_index];
        input_index++;
        output_index++;
    }
}
