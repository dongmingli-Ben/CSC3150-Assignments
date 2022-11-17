#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
	// init variables
	fs->volume = volume;

	// init constants
	fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
	fs->FCB_SIZE = FCB_SIZE;
	fs->FCB_ENTRIES = FCB_ENTRIES;
	fs->STORAGE_SIZE = VOLUME_SIZE;
	fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
	fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
	fs->MAX_FILE_NUM = MAX_FILE_NUM;
	fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
	fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

}


/*
Return the FCB index
*/
__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
	uchar * fcb;
	u32 block_id;
	u32 fcb_index;
	char * filename;
	fcb_index = fs_search_by_name(fs, s);
	if (fcb_index == 0x8000000) {
		// fcb not found, and no free fcb
		printf("Cannot create file %s due to OOM\n", s);
		assert(0);
	}
	fcb = fs_get_fcb(fs, fcb_index);
	filename = fcb_get_filename(fcb);
	// printf("%s - %s : %d\n", filename, s, my_strcmp(filename, s));
	if (my_strcmp(filename, s) != 0) {
		// not found
		if (op == G_READ) {
			printf("%s does not exist!\n", s);
			assert(0);
		}
		// create file if not exist
		assert(my_strlen(s) < 20);
		my_strcpy(filename, s);
		// allocate free blocks
		block_id = fs_search_freeblock(fs);
		fcb_get_filesize(fcb) = 0;  // return a reference
		fcb_get_start_block(fcb) = block_id;
		fs_set_superblock(fs, block_id, 1);  // 1 ==> used
		// increment global time
		gtime++;
		assert((gtime >> 16) == 0);
		// set modified time and created time
		fcb_get_modified_time(fcb) = gtime; // return a reference
		fcb_get_created_time(fcb) = gtime; // return a reference
	} 
	return fcb_index;
}

/*
fp: the corresponding FCB index
*/
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
	uchar * fcb;
	u32 filesize, block_id;
	fcb = fs_get_fcb(fs, fp);
	filesize = fcb_get_filesize(fcb);
	block_id = fcb_get_start_block(fcb);
	u32 pointer = fs->FILE_BASE_ADDRESS + block_id*(fs->STORAGE_BLOCK_SIZE);
	for (int i = 0; i < size; i++) {
		if (i >= filesize) {
			printf("Read EOF, stop reading\n");
			break;
		}
		output[i] = fs->volume[pointer+i];
	}
	// // set modified time
	// gtime++;
	// fcb_get_modified_time(fcb) = gtime;
}

/*
fp: the corresponding FCB index
*/
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
	assert(size <= (1 << 10));
	uchar * fcb;
	u32 block_id;
	fcb = fs_get_fcb(fs, fp);
	// clean up old content
	fs_rm_file_content(fs, fcb); // set bitmap (optional)
	fcb_get_filesize(fcb) = 0;
	// start writing data to file
	block_id = fcb_get_start_block(fcb);
	u32 pointer = fs->FILE_BASE_ADDRESS + block_id*(fs->STORAGE_BLOCK_SIZE);
	for (int i = 0; i < size; i++) {
		fs->volume[pointer+i] = input[i];
	}
	// setup fcb and super block
	fs_update_size(fs, fcb, size);
	// increment time
	gtime++;
	assert((gtime >> 16) == 0);
	fcb_get_modified_time(fcb) = gtime;
	return 0;
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	uchar *fcbs[1024];
	uchar * fcb;
	int file_num = 0;
	// loop over fcb
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		fcb = fs_get_fcb(fs, i);
		if (!fcb_is_valid(fcb)) {
			continue;
		}
		fcbs[file_num] = fcb;
		file_num++;
	}
	// sort
	// debug
	// for (int i = 0; i < file_num; i++) {
	// 	if (my_strcmp(fcb_get_filename(fcbs[i]), "EA\0") == 0) {
	// 		printf("--------DEBUG MESSAGE------------\n");
	// 		printf("%s %u\n", fcb_get_filename(fcbs[i]),
	// 			fcb_get_filesize(fcbs[i]));
	// 		printf("--------DEBUG MESSAGE------------\n");
	// 	}
	// }
	// bubble sort
	uchar * tmp;
	bool swap = false;
	for (int i = 0; i < file_num; i++) {
		for (int j = 0; j < file_num-1-i; j++) {
			swap = false;
			if (op == LS_D) {
				if (fcb_get_modified_time(fcbs[j]) < fcb_get_modified_time(fcbs[j+1])) {
					swap = true;
				} else if (fcb_get_modified_time(fcbs[j]) == fcb_get_modified_time(fcbs[j+1])
						&& fcb_get_filesize(fcbs[j]) < fcb_get_filesize(fcbs[j+1])) {
					swap = true;
				}
			} else if (op == LS_S) {
				if (fcb_get_filesize(fcbs[j]) < fcb_get_filesize(fcbs[j+1])) {
					swap = true;
				} else if (fcb_get_filesize(fcbs[j]) == fcb_get_filesize(fcbs[j+1])
						&& fcb_get_created_time(fcbs[j]) > fcb_get_created_time(fcbs[j+1])) {
					swap = true;
				}
			} else {
				printf("Option %d not recognized!\n", op);
				assert(0);
			}
			if (swap) {
				tmp = fcbs[j];
				fcbs[j] = fcbs[j+1];
				fcbs[j+1] = tmp;
			}
		}
	}
	// print to console
	if (op == LS_D) {
		printf("===sort by modified time===\n");
	} else {
		printf("===sort by file size===\n");
	}
	for (int i = 0; i < file_num; i++) {
		if (op == LS_D) {
			printf("%s\n", fcb_get_filename(fcbs[i]));
		} else {
			printf("%s %u\n", fcb_get_filename(fcbs[i]), fcb_get_filesize(fcbs[i]));
		}
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	assert(op == RM);
	uchar * fcb;
	u32 fcb_index;
	fcb_index = fs_search_by_name(fs, s);
	fcb = fs_get_fcb(fs, fcb_index);
	// set bit map
	fs_rm_file_content(fs, fcb);
	// remove fcb
	memset(fcb, 0, 32);
}

/*
Return pointer to the corresponding FCB entry.
No check is performed.
*/
__device__ uchar * fs_get_fcb(FileSystem *fs, u32 index) {
	return &(fs->volume[fs->SUPERBLOCK_SIZE + index*fs->FCB_SIZE]);
}

__device__ const uchar * fs_get_fcb(const FileSystem *fs, u32 index) {
	return &(fs->volume[fs->SUPERBLOCK_SIZE + index*fs->FCB_SIZE]);
}

/*
Update the bit map and the length attribute in FCB according to the 
new size of the file.
If the size is 0, one block will be used.
The corresponding bit map for the old file should be cleared before
calling this function (optional).

It will set the bit map for the consecutive 32 blocks (a little overhead)
*/
__device__ void fs_update_size(FileSystem *fs, uchar *fcb, u32 size) {
	u32 block_id, length;
	fcb_get_filesize(fcb) = size;
	block_id = fcb_get_start_block(fcb);
	assert(block_id % 32 == 0);  // blocks of every file should begin with 32x
	length = size / fs->STORAGE_BLOCK_SIZE;
	if (size % fs->STORAGE_BLOCK_SIZE) {
		length++;
	}
	memset(&(fs->volume[block_id/8]), 0, 4);
	fs_set_superblock(fs, block_id, 1);  // the first block should be marked as used even for 0 byte file
	for (u32 i = 1; i < length; i++) {
		fs_set_superblock(fs, block_id+i, 1);
	}
}

/*
Set the corresponding blocks in super block to be free
*/
__device__ void fs_rm_file_content(FileSystem *fs, uchar *fcb) {
	u32 block_id, size, length;
	block_id = fcb_get_start_block(fcb);
	size = fcb_get_filesize(fcb);
	if (size > 0) {
		length = size / fs->STORAGE_BLOCK_SIZE;
		if (size % fs->STORAGE_BLOCK_SIZE) {
			length++;
		}
	} else {
		length = 1;
	}
	for (u32 i = 0; i < length; i++) {
		fs_set_superblock(fs, block_id+i, 0);
	}
}

/*
Return true if the fcb entry is valid/used.

Entry = 0 ==> free/not used
*/
__device__ bool fcb_is_valid(const uchar * fcb) {
	for (int i = 0; i < 32; i++) {
		if (fcb[i] != 0) {
			return true;
		}
	}
	return false;
}

/*
Return a reference to the modified time
*/
__device__ unsigned short & fcb_get_modified_time(const uchar * fcb) {
	u32 * ints = (u32 *) &(fcb[20]);
	unsigned short * shorts = (unsigned short *) &(ints[2]);
	return shorts[0];
}

/*
Return a reference to the created time
*/
__device__ unsigned short & fcb_get_created_time(const uchar * fcb) {
	u32 * ints = (u32 *) &(fcb[20]);
	unsigned short * shorts = (unsigned short *) &(ints[2]);
	return shorts[1];
}

/*
Return a reference to the start block id
*/
__device__ u32 & fcb_get_start_block(const uchar * fcb) {
	u32 * ints = (u32 *) &(fcb[20]);
	return ints[0];
}

/*
Get the file name from the FCB
*/
__device__ char * fcb_get_filename(const uchar * fcb) {
	return (char *) fcb;
}

/*
Return a reference to the file size
*/
__device__ u32 & fcb_get_filesize(const uchar * fcb) {
	u32 * ints = (u32 *) &(fcb[20]);
	return ints[1];
}

/*
Search for a match entry in FCB.
If no match is found, return a free FCB
*/
__device__ u32 fs_search_by_name(const FileSystem *fs, const char *s) {
	u32 fcb_index = 0;
	u32 free_fcb = 0x80000000;
	const uchar * fcb;
	char * name;
	for (fcb_index = 0; fcb_index < fs->FCB_ENTRIES; fcb_index++) {
		fcb = fs_get_fcb(fs, fcb_index);
		if (!fcb_is_valid(fcb)) {
			free_fcb = fcb_index;
			continue;
		}
		name = fcb_get_filename(fcb);
		if (my_strcmp(name, s) == 0) {
			return fcb_index;
		}
	}
	return free_fcb;
}

/*
Set bitmap

1: used, 0: free
*/
__device__ void fs_set_superblock(FileSystem *fs, u32 block_id, int op) {
	uchar higher, lower;
	uchar position = block_id % 8;
	higher = (fs->volume[block_id/8] >> (8-position)) << (8-position);
	lower = ((fs->volume[block_id/8] << (position+1)) & 0xff) >> (position+1);
	fs->volume[block_id/8] = higher | lower | (op << (7-position));
}

/*
Search the bitmap to find a free block.
If no free block available, return 0x8000000.

The bitmap is big endian, i.e. higher bits stores blocks in the front.

Since max file size is 1KB = 32 blocks, search free blocks every other 32 blocks
*/
__device__ u32 fs_search_freeblock(const FileSystem *fs) {
	uchar val;
	for (u32 i = 0; i < fs->SUPERBLOCK_SIZE; i+=4) {
		val = fs->volume[i];
		if ((val >> 7) == 0) {
			return 8*i;
		}
		// idx++;
		// if ((val ^ 0xbf) == 0) {
		// 	return idx;
		// }
		// idx++;
		// if ((val ^ 0xdf) == 0) {
		// 	return idx;
		// }
		// idx++;
		// if ((val ^ 0xef) == 0) {
		// 	return idx;
		// }
		// idx++;
		// if ((val ^ 0xf7) == 0) {
		// 	return idx;
		// }
		// idx++;
		// if ((val ^ 0xfb) == 0) {
		// 	return idx;
		// }
		// idx++;
		// if ((val ^ 0xfd) == 0) {
		// 	return idx;
		// }
		// idx++;
		// if ((val ^ 0xfe) == 0) {
		// 	return idx;
		// }
		// idx++;
	}
	return 0x8000000;
}

// some standard C library functions

__device__ char *(my_strcpy)(char *s1, const char *s2)
	{	/* copy char s2[] to s1[] */
	char *s = s1;

	for (s = s1; (*s++ = *s2++) != '\0'; )
		;
	return (s1);
	}

__device__ size_t (my_strlen)(const char *s)
	{	/* find length of s[] */
	const char *sc;

	for (sc = s; *sc != '\0'; ++sc)
		;
	return (sc - s);
	}

/*
0: equal, 1 or -1: not equal
*/
__device__ int (my_strcmp)(const char *s1, const char *s2)
	{	/* compare unsigned char s1[], s2[] */
	// printf("%s -- %s\n", s1, s2);
	for (; *s1 == *s2; ++s1, ++s2)
		if (*s1 == '\0')
			return (0);
	return (*(unsigned char *)s1 < *(unsigned char *)s2
		? -1 : +1);
	}