#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void print_page_table(VirtualMemory *vm) {
  u32 entry;
  for (u32 i = 0; i < vm->PAGE_ENTRIES; i++) {
    entry = vm->invert_page_table[i];
    printf("Physical mem frame %u stores logical page %u, counter %u\n", i, entry>>11, entry & 0x7ff);
  }
  for (u32 i = 0; i < vm->SWAP_ENTRIES; i++) {
    printf("Swap mem frame %u stores logical page %u\n", i, vm->swap_table[i]);
  }
}

// __device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
//                                                          int input_size) {
//     // print_page_table(vm);
//     for (int i = 0; i < input_size; i++) {
//         vm_write(vm, i, input[i]);
//     }
//     printf("Finish writing\n");
//     for (int i = input_size - 1; i >= input_size - 32769; i--)
//         int value = vm_read(vm, i);
//     printf("Finish reading\n");

//     vm_snapshot(vm, results, 0, input_size);
//     printf("Finish snapshot\n");
// }

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
    int input_size) {
    // write the data.bin to the VM starting from address 32*1024
    for (int i = 0; i < input_size; i++)
        vm_write(vm, 32*1024+i, input[i]);
    printf("Finish writing, page fault num %d\n", *vm->pagefault_num_ptr);
    // print_page_table(vm);
    // write (32KB-32B) data  to the VM starting from 0
    for (int i = 0; i < 32*1023; i++)
        vm_write(vm, i, input[i+32*1024]);
    printf("Finish writing, page fault num %d\n", *vm->pagefault_num_ptr);
    // print_page_table(vm);
    // readout VM[32K, 160K] and output to snapshot.bin, which should be the same with data.bin
    vm_snapshot(vm, results, 32*1024, input_size);
    printf("Finish snapshot, page fault num %d\n", *vm->pagefault_num_ptr);
    // print_page_table(vm);
}