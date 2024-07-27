#pragma once 

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

// https://stackoverflow.com/a/24336429
const char *cl_errstr(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

// Print error message if error is not CL_SUCCESS
void cl_print_err(const char* entry, cl_int error){
    if (error == CL_SUCCESS)
        return;

    char buffer[128];
    sprintf(buffer, "%-50s%s", entry, cl_errstr(error));
    std::cout << buffer << std::endl;
}

struct CL{
    cl_int           device_count;
    cl_device_id*    devices = nullptr;
    cl_context       context;
    cl_command_queue command_queue;
    cl_program       program;

    size_t           max_workgroup_size;

    std::vector<cl_kernel>   kernels;

    void free(){
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        clReleaseProgram(program);

        for (cl_kernel kernel : kernels)
            clReleaseKernel(kernel);

        delete[] devices;
    }   

    void init(const char* path, std::vector<const char*> kernels){
        // Get platform and device info 
        cl_platform_id* platforms = NULL;
        cl_uint         num_platforms;

        // Set up the platform
        cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
        cl_print_err("Get platform count:\t", status);

        platforms = new cl_platform_id[num_platforms]();
            
        status = clGetPlatformIDs(num_platforms, platforms, NULL);
        cl_print_err("Get platform ids:\t", status);

        // Get the devices list and choose the device
        cl_uint num_devices;

        status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL,
            &num_devices);
        cl_print_err("Get device count:\t", status);

        devices = new cl_device_id[num_devices]();
    
        status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices,
            devices, NULL);
        cl_print_err("Get device ids:\t\t", status);

        status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, 
            sizeof(max_workgroup_size), &max_workgroup_size, nullptr);
        cl_print_err("Get max work group size:\t", status);

        // Create a OpenCL context for each device
        context = clCreateContext(NULL, num_devices, devices, NULL, 
            NULL, &status);
        cl_print_err("Context creation:\t", status);

        // Enable profiling setting, this is incompatible with out of order 
        cl_queue_properties prop[] = 
            { CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 };
            
        // Create a command buffer with out of order execution
        command_queue = clCreateCommandQueueWithProperties(
            context, devices[0], prop, &status);
        cl_print_err("Command queue creation:\t", status);

        // Load kernel code from file
        std::ifstream f(path);
        if (!f.is_open()){
            std::cout << "Could not open file: " << path << std::endl;
            return;
        }
        
        std::stringstream ssbuffer;
        ssbuffer << f.rdbuf();

        std::string str         = ssbuffer.str();
        const char* kernel_code = str.data();
            
        // Build program from source code at start
        program = clCreateProgramWithSource(
            context, 1, &kernel_code, NULL, &status);
        cl_print_err("Progam creation:\t", status);

        status = clBuildProgram(
            program, 1, devices, NULL, NULL, NULL);
        cl_print_err("Program build:\t\t", status);

        if (status != CL_SUCCESS){
            size_t logsize;
            clGetProgramBuildInfo(program, devices[0], 
            CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsize);
            
            char* plog = new char[logsize];
            clGetProgramBuildInfo(program, devices[0], 
            CL_PROGRAM_BUILD_LOG, logsize, plog, NULL);
            
            std::cout << plog;
            delete[] plog;
        }

        char buffer[128];
        for (const char* kernel : kernels){
            cl_int status;

            this->kernels.push_back(clCreateKernel(program, kernel, &status));
            
            sprintf(buffer, "%s kernel creation", kernel);
            cl_print_err(buffer, status);
        }

        delete[] platforms;
    }

    // Allocate a cl_mem object and handle errors
    cl_mem alloc_buffer(
        const char* name, int size, void* data, cl_mem_flags flag){

        cl_int status;
        cl_mem buffer = clCreateBuffer(context, flag, size, data, &status);
        
        char str[128];
        sprintf(str, "Alloc buffer %s", name);
        cl_print_err(str, status);
        
        return buffer;
    }

    double time_execution(int kernel, cl_uint work_dim, 
        size_t* global_work_size, size_t* local_work_size){

        cl_int   status;
        cl_event event;

        status = clEnqueueNDRangeKernel(command_queue, kernels[kernel], 
            work_dim, NULL, global_work_size, local_work_size, 0, 
            NULL, &event);
        cl_print_err("Enqueue kernel:\t", status);

        status = clWaitForEvents(1, &event);
        cl_print_err("Wait for event:\t", status);

        cl_ulong start, end;

        status = clGetEventProfilingInfo(
            event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        cl_print_err("Get event start:\t", status);

        status = clGetEventProfilingInfo(
            event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        cl_print_err("Get event end:\t", status);

        double nanoseconds = (end - start) * 1e-6;

        clReleaseEvent(event);

        return nanoseconds;
    }

    void set_arg(int kernel, int index, size_t size, const void* arg){
        cl_int status = clSetKernelArg(kernels[kernel], index, size, arg);
        cl_print_err("Set arg:\t", status);
    }
};

