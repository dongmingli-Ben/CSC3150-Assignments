﻿#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>


#ifdef TEST1
__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                                                         int input_size) {
    for (int i = 0; i < input_size; i++) {
        vm_write(vm, i, input[i]);
    }
    // printf("Finish writing\n");
    // print_page_table(vm);
    for (int i = input_size - 1; i >= input_size - 32769; i--)
        int value = vm_read(vm, i);
    // printf("Finish reading\n");

    vm_snapshot(vm, results, 0, input_size);
    // printf("Finish snapshot\n");
}
#else
__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
    int input_size) {
    // write the data.bin to the VM starting from address 32*1024
    for (int i = 0; i < input_size; i++)
        vm_write(vm, 32*1024+i, input[i]);
    // printf("Finish writing, page fault num %d\n", *vm->pagefault_num_ptr);
    // print_page_table(vm);
    // write (32KB-32B) data  to the VM starting from 0
    for (int i = 0; i < 32*1023; i++)
        vm_write(vm, i, input[i+32*1024]);
    // printf("Finish writing, page fault num %d\n", *vm->pagefault_num_ptr);
    // print_page_table(vm);
    // readout VM[32K, 160K] and output to snapshot.bin, which should be the same with data.bin
    vm_snapshot(vm, results, 32*1024, input_size);
    // printf("Finish snapshot, page fault num %d\n", *vm->pagefault_num_ptr);
    // print_page_table(vm);
}
#endif