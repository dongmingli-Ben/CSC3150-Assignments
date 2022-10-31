﻿#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void print_page_table(VirtualMemory *vm) {
  u32 entry;
  for (u32 i = 0; i < vm->PAGE_ENTRIES; i++) {
    entry = vm->invert_page_table[i];
    printf("Physical mem frame %u stores logical page %u, counter %u\n", i, entry>>16, entry & 0xffff);
  }
  for (u32 i = 0; i < vm->PAGE_ENTRIES*4; i++) {
    u32 index = i/2;
    entry = vm->invert_page_table[vm->PAGE_ENTRIES+index];
    if (i % 2 == 1) {
      // lower
      entry = entry & 0xffff;
    } else {
      entry = entry >> 16;
    }
    printf("Swap mem frame %u stores logical page %u\n", i, entry);
  }
}

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
  // print_page_table(vm);
  for (int i = 0; i < input_size; i++) {
    vm_write(vm, i, input[i]);
    // if ((i+1) % (1024*32) == 0) {
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //   print_page_table(vm);
    // }
  }
  // for (int i = 0; i < vm->PAGE_ENTRIES*5; i++) {
  //   u32 entry;
  //   if (i < 1024) {
  //     entry = vm->invert_page_table[i] >> 16;
  //   } else {
  //     entry = vm->invert_page_table[(i-vm->PAGE_ENTRIES)/2 + vm->PAGE_ENTRIES];
  //     if ((i-vm->PAGE_ENTRIES)%2 == 1) {
  //       // lower
  //       entry = entry & 0xffff;
  //     } else {
  //       entry = entry >> 16;
  //     }
  //   }
  //   // if (entry != i) {
  //     printf("physical page %d stores logical page %u\n", i, entry);
  //   // }
  // }
  // for (int i = 0; i < input_size; i++) {
  //   if (i >= 128*1024) {
  //     results[i] = vm->buffer[i-128*1024];
  //   } else {
  //     results[i] = vm->storage[i];
  //   }
  // }
  // return;
  printf("Finish writing\n");
  for (int i = input_size - 1; i >= input_size - 32769; i--)
    int value = vm_read(vm, i);
  printf("Finish reading\n");
  // print_page_table(vm);

  vm_snapshot(vm, results, 0, input_size);
  printf("Finish snapshot\n");
}
