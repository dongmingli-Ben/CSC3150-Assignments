﻿#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>

#ifdef VER3
// lock for page lookup and swapping
__shared__ int lock;
#endif

#ifdef DEBUG
// debug
__device__ void print_page_table_debug(VirtualMemory *vm) {
  u32 entry;
  for (u32 i = 0; i < vm->PAGE_ENTRIES; i++) {
    entry = vm->invert_page_table[i];
    printf("Physical mem frame %u stores logical page %u, counter %u, pid %u\n", 
        i, (entry>>11) & 0x1fff, entry & 0x7ff, entry >> 29);
  }
  for (u32 i = 0; i < vm->SWAP_ENTRIES; i++) {
    printf("Swap mem frame %u stores logical page %u, pid %u\n", 
        i, vm->swap_table[i] & 0x1fff, vm->swap_table[i] >> 13);
  }
}
#endif

__device__ void init_invert_page_table(VirtualMemory *vm) {

    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
        // vm->invert_page_table[i] = i << 16;
        // store the time when the physical frame is used
        // vm->invert_page_table[i + 2*vm->PAGE_ENTRIES] = 0;
    }
    // initialize swap table
    for (int i = 0; i < vm->SWAP_ENTRIES; i++) {
        vm->swap_table[i] = 0x80000000;
    }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, u32 *swap_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
    // init variables
    vm->buffer = buffer;
    vm->storage = storage;
    vm->invert_page_table = invert_page_table;
    vm->swap_table = swap_table;
    vm->pagefault_num_ptr = pagefault_num_ptr;

    // init constants
    vm->PAGESIZE = PAGESIZE;
    vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
    vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
    vm->STORAGE_SIZE = STORAGE_SIZE;
    vm->PAGE_ENTRIES = PAGE_ENTRIES;
    vm->SWAP_ENTRIES = STORAGE_SIZE / PAGESIZE;

    // before first vm_write or vm_read
    init_invert_page_table(vm);
}

/*
Map a logical address to a physical one.
Will swap if it is necessary.
If write is set to false, it is read mode. It will raise a segmentation fault
if the page is not used (written) yet.
Note: virtual page range 2^13, pid range 2^2
*/
__device__ u32 vm_map_physical(VirtualMemory *vm, u32 addr, bool write) {
    u32 PAGE_BIT = 13;
    u32 FRAME_BIT = 11;
    u32 PAGE_MASK = 0x1fff;
    u32 COUNTER_MASK = 0x7ff;
    assert(PAGE_BIT + FRAME_BIT < 29); // otherwise u32 is not enough to hold an entry
#ifdef VER2
    u32 pid = 0;
#else
    u32 pid = threadIdx.x;
#endif
    u32 entry = (((addr >> 5) & PAGE_MASK) << FRAME_BIT) | (pid << 29);
    u32 phy_addr;
    /* search in inverted page table */
    // also search for the LRU physical frame
    uchar tmp; // used as temporary to swap data
    u32 time = 0;
    u32 victim_frame = 0;
    u32 free_swap_frame = 1 << 31;
    bool found_page = false;
    bool found_free = false;
    bool found_free_swap = false;
    bool new_phy_frame = false;  // if it is false, do not increment the counter
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        if ((vm->invert_page_table[i] >> FRAME_BIT) == (entry >> FRAME_BIT)) {
            // found page
            phy_addr = (i << 5) | (addr & 0x1f);
            found_page = true;
            // set counter
            if ((vm->invert_page_table[i] & COUNTER_MASK) > 1) {
                // use a new frame (last active frame different than this one)
                new_phy_frame = true;
                vm->invert_page_table[i] &= (~COUNTER_MASK); // set the counter to 0, so it can be increamented to 1 at the end
            } else {
                vm->invert_page_table[i] = entry + 1;
            }
            break;
        }
        if (!found_free) {
            if ((vm->invert_page_table[i] >> 31) == 1) {
                victim_frame = i;
                found_free = true;
                continue;
            }
            if ((vm->invert_page_table[i] & COUNTER_MASK) > time) {
                time = vm->invert_page_table[i] & COUNTER_MASK;
                victim_frame = i;
            }
        }
    }
    if (found_page == false) {
        // page fault
        // (*vm->pagefault_num_ptr)++;
        atomicAdd(vm->pagefault_num_ptr, 1);
        // swap page (page at victim frame) from storage
        new_phy_frame = true;
        for (int i = 0; i < vm->SWAP_ENTRIES; i++) {
            if ((vm->swap_table[i] >> 31) == 1) {
                // swap frame not used yet
                if (found_free_swap) continue;
                found_free_swap = true;
                free_swap_frame = i;
                continue;
            }
            if ((!found_page) && (vm->swap_table[i] == (entry >> FRAME_BIT))) {
                // found corresponding page in disk
                found_page = true;
                if (found_free) {
                    // use the free frame, no swap
                    for (int j = 0; j < 32; j++) {
                        vm->buffer[32*victim_frame+j] = vm->storage[j];
                    }
                } else {
                    // replace LRU page
                    // move victim data into storage
                    // move tmp data into mem
                    for (int j = 0; j < 32; j++) {
                        tmp = vm->storage[32*i+j];
                        vm->storage[32*i+j] = vm->buffer[32*victim_frame+j];
                        vm->buffer[32*victim_frame+j] = tmp;
                    }
                    // modify page table to manage swap entry
                    vm->swap_table[i] = vm->invert_page_table[victim_frame] >> FRAME_BIT;
                }
                vm->invert_page_table[victim_frame] = entry;
                // set physical address
                phy_addr = (victim_frame << 5) | (addr & 0x1f);
                // break;
            }
        }
    }
    if (!found_page && !write) {
        // invalid page (haven't been written)
        printf("page num %u, addr %u, entry %u haven't been used yet, cannot read, segmentation fault\n", 
            addr >> 5, addr, entry >> FRAME_BIT);
#ifdef DEBUG
        print_page_table_debug(vm);
#endif
        assert(0);
    } else if (!found_page) {
        // write mode, use free swap frame to hold victim page
        if ((free_swap_frame >> 31) == 1) {
            printf("swap memory used up\n");
#ifdef DEBUG
            print_page_table_debug(vm);
#endif
            assert(0);
        }
        if (!found_free) {
            for (int j = 0; j < 32; j++) {
                vm->storage[32*free_swap_frame+j] = vm->buffer[32*victim_frame+j];
            }
            // modify page table
            vm->swap_table[free_swap_frame] = vm->invert_page_table[victim_frame] >> FRAME_BIT;
        }
        vm->invert_page_table[victim_frame] = entry;
        phy_addr = (victim_frame << 5) | (addr & 0x1f);
    }
    // increment time counter
    if (new_phy_frame) {
        for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
            if ((vm->invert_page_table[i] >> 31) == 0)
                vm->invert_page_table[i]++;
        }
    }
    return phy_addr;
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
    /* Complate vm_read function to read single element from data buffer */
#ifdef VER1
    if ((addr>>5)%4 == threadIdx.x) {
        u32 phy_addr = vm_map_physical(vm, addr, true);
        return vm->buffer[phy_addr];
    }
    return 0;
#endif
#ifdef VER2
    u32 phy_addr = vm_map_physical(vm, addr, false);
    return vm->buffer[phy_addr]; //TODO
#endif
#ifdef VER3
    u32 phy_addr;
    // critical section
    lock = 0;
    bool acquire_lock = false;
    while (!acquire_lock) {
        if (atomicCAS(&lock, 0, 1) == 0) {
            phy_addr = vm_map_physical(vm, addr, false);
            // release the lock
            acquire_lock = true;
            atomicExch(&lock, 0);
        }
    }
    return vm->buffer[phy_addr];
#endif
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
    /* Complete vm_write function to write value into data buffer */
#ifdef VER1
    if ((addr>>5)%4 == threadIdx.x) {
        u32 phy_addr = vm_map_physical(vm, addr, true);
        vm->buffer[phy_addr] = value;
    }
#endif
#ifdef VER2
    u32 phy_addr = vm_map_physical(vm, addr, true);
    vm->buffer[phy_addr] = value;
#endif
#ifdef VER3
    u32 phy_addr;
    // critical section
    bool acquire_lock = false;
    while (!acquire_lock) {
        if (atomicCAS(&lock, 0, 1) == 0) {
            phy_addr = vm_map_physical(vm, addr, true);
            // release the lock
            acquire_lock = true;
            atomicExch(&lock, 0);
        }
    }
    vm->buffer[phy_addr] = value;
#endif
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                                                        int input_size) {
    /* Complete snapshot function togther with vm_read to load elements from data
     * to result buffer */
    for (int i = 0; i < input_size; i++) {
#ifdef VER1
        if ((i>>5)%4 == threadIdx.x) {
            results[i] = vm_read(vm, i+offset);
        }
#endif
#ifdef VER2
        results[i] = vm_read(vm, i+offset);
#endif
#ifdef VER3
        if (threadIdx.x == 0) {
            results[i] = vm_read(vm, i+offset);
        }
#endif
    }
}

