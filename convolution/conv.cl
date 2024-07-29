// #define FILTERW  5 
// #define FILTERH  5
// #define CHANNELS 3
// #define DEBUG

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
                // printf("conv[%d] += %f * %f(%d * %d)\n", result_index, values[filterc + c], filters[filters_index], filterc + c, filters_index);
                filters_index++;
            }
        }
    }

    result[result_index] = acc;

#ifdef DEBUG
    printf("conv[%d]: %f\n", result_index, acc);
#endif
}

__kernel
void convolution_opt(
__global float* filters,
__global float* values,
__global float* result){

    int filtery = get_global_id(0);
    int filterx = get_global_id(1);
    int feature = get_global_id(2);

    int filtersy = get_global_size(0) / BSIZE;
    int filtersx = get_global_size(1);
    int features = get_global_size(2);

    int batch = filtery / filtersy;

    int result_index = (filtery * filtersx + filterx) * features + feature;

    filtery %= filtersy;

    // Index of filter being applied from filter list
    int filters_index = feature * FILTERH * FILTERW * CHANNELS;

    int image_start = batch * PREVH * PREVW;
    // Move to start of filter within batch
    image_start += filtery * STRIDEY * PREVW + filterx * STRIDEX;

    int unit_index = (filtery * filtersx + filterx) * features + feature;

    float acc = 0.0f;
    
    #pragma unroll
    for (int y = 0; y < FILTERH; y++){
        int filtery = image_start + y * PREVW;

        #pragma unroll
        for (int x = 0; x < FILTERW; x++){
            int filterx = filtery + x;
            int filterc = filterx * CHANNELS;

            #pragma unroll
            for (int c = 0; c < CHANNELS; c++){
                acc += values[filterc + c] * filters[filters_index];
                // printf("conv[%d] += %f * %f(%d * %d)\n", result_index, values[filterc + c], filters[filters_index], filterc + c, filters_index);
                filters_index++;
            }
        }
    }

    result[result_index] = acc;

#ifdef DEBUG
    printf("conv[%d]: %f\n", result_index, acc);
#endif
}

float convolve_channels(
__local    float* values_cache,
__constant float* filters,
  const    int    channels){

    float acc = 0.0f;
    for (int c = 0; c < channels; c++){
        acc += values_cache[c] * filters[c];
    }

    return acc;
}

// Local cache of values for each filter position is not faster
// than the naive approach becuase of the work group barrier to sync
// the cache filling

// Global size: [bsize * filtersy, filtersx, features]
// Local size:  [1, 1, features]
__kernel
void convolution_cache(
  const    int    bsize,
  const    int    prevw,
  const    int    prevh,
  const    int    stridex,
  const    int    stridey,
__constant float* values,
__constant float* filters,
__global   float* result){

    int filtery = get_global_id(0);
    int filterx = get_global_id(1);
    int feature = get_global_id(2);

    int filtersy = get_global_size(0) / bsize;
    int filtersx = get_global_size(1);
    int features = get_global_size(2);

    int batch = filtery / filtersy;

    int result_index = (filtery * filtersx + filterx) * features + feature;

    filtery %= filtersy;

    // Fill chache for a single filter position
    // A single work goup will fill the cache for a single filter position
    // split between feature's work items
    int cache_size = FILTERH * FILTERW * CHANNELS;

    __local float values_cache[FILTERH][FILTERW][CHANNELS];

    int image_start = batch * prevh * prevw;
    int mask_start  = image_start + filtery * stridey * prevw + filterx * stridex;
    
    for (int i = feature; i < cache_size; i += features){
        int c = i % CHANNELS;
        int x = (i / CHANNELS) % FILTERH;
        int y = i / (CHANNELS * FILTERW);

        int image_index = (mask_start + y * prevw + x) * CHANNELS + c;

        values_cache[y][x][c] = values[image_index];
    }

    // Sync work group to make sure cache is filled
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // Chache size is the same as filter size
    int filter_index = cache_size * feature;

    float acc = 0.0f;
    for (int y = 0; y < FILTERH; y++){
        for (int x = 0; x < FILTERW; x++){
            for (int c = 0; c < CHANNELS; c++){
                acc += values_cache[y][x][c] * filters[filter_index];
                filter_index++;
            }
        }
    }

    result[result_index] = acc;

#ifdef DEBUG
    printf("conv[%d]: %f\n", result_index, acc);
#endif
}
