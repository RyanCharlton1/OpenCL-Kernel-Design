// Orginal, working kernel

// Filters: features[filterh[filterw[channels]]]
// Values:  batches[prevh[prevw[channels]]]
// result:  batches[filtersy[filtersx[features]]]
__kernel
void convolution(
         int    prevw,
         int    prevh,
         int    filtersy,
         int    filterw,
         int    filterh,
         int    channels,
         int    stridex,
         int    stridey,
__global float* filters,
__global float* values,
__global float* result){

    int filtery = get_global_id(0);
    int filterx = get_global_id(1);
    int feature = get_global_id(2);

    int filtersx = get_global_size(1);
    int features = get_global_size(2);

    int batch = filtery / filtersy;
    filtery %= filtersy;

    // Index of filter being applied from filter list
    int filters_index = feature * filterh * filterw * channels;

    int image_start;
    // Start at batch's(image's) 0,0
    image_start  = batch * prevh * prevw ;
    // Move to start of filter within batch
    int filter_start;
    filter_start  = image_start;
    filter_start += filtery * stridey * prevw + filterx * stridex;

    int unit_index = (filtery * filtersx + filterx) * features + feature;

    // Index of output node being calculated
    int result_index;
    result_index  = batch * filtersy * filtersx;
    result_index += filtery * filtersx;
    result_index += filterx;
    result_index *= features;
    result_index += feature;

    float acc = 0.0f;
    for (int y = 0; y < filterh; y++){
        int filtery = filter_start + y * prevw;

        for (int x = 0; x < filterw; x++){
            int filterx = filtery + x;
            int filterc = filterx * channels;

            for (int c = 0; c < channels; c++){
                acc += values[filterc + c] * filters[filters_index];
                //printf("conv[%d] += %f * %f(%d * %d)\n", result_index, values[filterc + c], filters[filters_index], filterc + c, filters_index);
                filters_index++;
            }
        }
    }

    result[result_index] = acc;

#ifdef DEBUG
    printf("conv[%d]: %f\n", result_index, acc);
#endif
}
