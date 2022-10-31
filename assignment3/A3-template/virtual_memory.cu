#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
// debug
__device__ void print_page_table_debug(VirtualMemory *vm) {
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

__device__ void init_invert_page_table(VirtualMemory *vm) {

    for (int i = 0; i < 4*(vm->PAGE_ENTRIES); i++) {
        vm->invert_page_table[i] = 0x80008000; // invalid := MSB is 1
        // vm->invert_page_table[i] = i << 16;
        // store the time when the physical frame is used
        // vm->invert_page_table[i + 2*vm->PAGE_ENTRIES] = 0;
    }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
    // init variables
    vm->buffer = buffer;
    vm->storage = storage;
    vm->invert_page_table = invert_page_table;
    vm->pagefault_num_ptr = pagefault_num_ptr;

    // init constants
    vm->PAGESIZE = PAGESIZE;
    vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
    vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
    vm->STORAGE_SIZE = STORAGE_SIZE;
    vm->PAGE_ENTRIES = PAGE_ENTRIES;

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
    u32 pid = 0;
    u32 entry = (((addr >> 5) & 0x1fff) << 16) | (pid << 29);
    u32 phy_addr;
    /* search in inverted page table */
    // also search for the LRU physical frame
    u32 time = 0;
    u32 victim_frame = 0;
    u32 free_swap_frame = 1 << 31;
    bool found_page = false;
    bool found_free = false;
    bool found_free_swap = false;
    bool new_phy_frame = false;  // if it is false, do not increment the counter
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        if ((vm->invert_page_table[i] >> 16) == (entry >> 16)) {
            // found page
            phy_addr = (i << 5) | (addr & 0x1f);
            found_page = true;
            // set counter
            if ((vm->invert_page_table[i] & 0xffff) > 1) {
                // use a new frame
                new_phy_frame = true;
                vm->invert_page_table[i] &= 0xffff0000; // set the counter to 0, so it can be increamented to 1 at the end
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
            if ((vm->invert_page_table[i] & 0xffff) > time) {
                time = vm->invert_page_table[i] & 0xffff;
                victim_frame = i;
            }
        }
    }
    if (found_page == false) {
        // page fault
        (*vm->pagefault_num_ptr)++;
        // swap page (page at victim frame) from storage
        if ((vm->invert_page_table[victim_frame] & 0xffff) > 1) {
            new_phy_frame = true;
        }
        u32 pt_entry;
        for (int i = 0; i < 4*(vm->PAGE_ENTRIES); i++) {
            if (i%2 == 1) {
                // lower 16 bits
                pt_entry = vm->invert_page_table[i/2 + vm->PAGE_ENTRIES] & 0xffff;
            } else {
                // higher 16 bits
                pt_entry = vm->invert_page_table[i/2 + vm->PAGE_ENTRIES] >> 16;
            }
            if ((pt_entry >> 15) == 1) {
                // swap frame not used yet
                if (found_free_swap) continue;
                found_free_swap = true;
                free_swap_frame = i;
                continue;
            }
            if ((!found_page) && (pt_entry == (entry >> 16))) {
                // found corresponding page in disk
                found_page = true;
                if (found_free) {
                    // use the free frame, no swap
                    for (int j = 0; j < 32; j++) {
                        vm->buffer[32*victim_frame+j] = vm->storage[j];
                    }
                } else {
                    // replace LRU page
                    uchar tmp_data[32];  // temporary physical frame data
                    for (int j = 0; j < 32; j++) {
                        tmp_data[j] = vm->storage[32*i+j];
                    }
                    // move victim data into storage
                    for (int j = 0; j < 32; j++) {
                        vm->storage[32*i+j] = vm->buffer[32*victim_frame+j];
                    }
                    // move tmp data into mem
                    for (int j = 0; j < 32; j++) {
                        vm->buffer[32*victim_frame+j] = tmp_data[j];
                    }
                    // modify page table to manage swap entry
                    if (i%2 == 1) {
                        // lower 16 bits
                        vm->invert_page_table[i/2 + vm->PAGE_ENTRIES] = 
                            (vm->invert_page_table[victim_frame] >> 16) | 
                            (vm->invert_page_table[i/2 + vm->PAGE_ENTRIES] & 0xffff0000);
                    } else {
                        // higher 16 bits
                        vm->invert_page_table[i/2 + vm->PAGE_ENTRIES] = 
                            (vm->invert_page_table[victim_frame] & 0xffff0000) | 
                            (vm->invert_page_table[i/2 + vm->PAGE_ENTRIES] & 0xffff);
                    }
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
            addr >> 5, addr, entry >> 16);
        print_page_table_debug(vm);
        assert(0);
    } else if (!found_page) {
        // write mode, use free swap frame to hold victim page
        if ((free_swap_frame >> 31) == 1) {
            printf("swap memory used up\n");
            assert(0);
        }
        if (!found_free) {
            // move victim data into storage
            // printf("Before swap op, victim frame %u, entry %u\n", 
            //     victim_frame,
            //     vm->invert_page_table[victim_frame] >> 16);
            for (int j = 0; j < 32; j++) {
                vm->storage[32*free_swap_frame+j] = vm->buffer[32*victim_frame+j];
            }
            // modify page table
            if (free_swap_frame%2 == 1) {
                // lower 16 bits
                // printf("Before swap op, swap frame %u, entry %u\n", 
                //     free_swap_frame,
                //     vm->invert_page_table[free_swap_frame/2 + vm->PAGE_ENTRIES] & 0xffff);
                vm->invert_page_table[free_swap_frame/2 + vm->PAGE_ENTRIES] = 
                    (vm->invert_page_table[victim_frame] >> 16) |
                    (vm->invert_page_table[free_swap_frame/2 + vm->PAGE_ENTRIES] & 0xffff0000);
                // printf("After swap op, swap frame %u, entry %u\n", 
                //     free_swap_frame,
                //     vm->invert_page_table[free_swap_frame/2 + vm->PAGE_ENTRIES] & 0xffff);
            } else {
                // higher 16 bits
                // printf("Before swap op, swap frame %u, entry %u\n", 
                //     free_swap_frame,
                //     vm->invert_page_table[free_swap_frame/2 + vm->PAGE_ENTRIES] >> 16);
                vm->invert_page_table[free_swap_frame/2 + vm->PAGE_ENTRIES] = 
                    (vm->invert_page_table[victim_frame] & 0xffff0000) |
                    (vm->invert_page_table[free_swap_frame/2 + vm->PAGE_ENTRIES] & 0xffff);
                // printf("After swap op, swap frame %u, entry %u\n", 
                //     free_swap_frame,
                //     vm->invert_page_table[free_swap_frame/2 + vm->PAGE_ENTRIES] >> 16);
            }
        }
        vm->invert_page_table[victim_frame] = entry;
        // printf("After swap op, victim frame %u, entry %u\n", 
        //     victim_frame,
        //     vm->invert_page_table[victim_frame] >> 16);
        phy_addr = (victim_frame << 5) | (addr & 0x1f);
        // printf("Swap physical fram %u to swap frame %u, ", victim_frame, free_swap_frame);
        // printf("logical frame at %u is %u\n", victim_frame, entry>>16);
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
    u32 phy_addr = vm_map_physical(vm, addr, false);
    return vm->buffer[phy_addr]; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
    /* Complete vm_write function to write value into data buffer */
    u32 phy_addr = vm_map_physical(vm, addr, true);
    vm->buffer[phy_addr] = value;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                                                        int input_size) {
    /* Complete snapshot function togther with vm_read to load elements from data
     * to result buffer */
    for (int i = offset; i < input_size; i++) {
        results[i-offset] = vm_read(vm, i);
    }
}

