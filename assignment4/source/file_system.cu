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
	if (fcb_index == 0x80000000) {
		// fcb not found, and no free fcb
		printf("Cannot create file %s due to OOM\n", s);
		assert(0);
	}
	fcb = fs_get_fcb(fs, fcb_index);
	filename = fcb_get_filename(fcb);
	if (my_strcmp(filename, s) != 0) {
		// not found
		if (op == G_READ) {
			printf("%s does not exist!\n", s);
			assert(0);
		}
		// create file if not exist
		assert(my_strlen(s) < 20);
		my_strcpy(filename, s);
		// do not need to allocate free blocks
		fcb_get_filesize(fcb) = 0;  // return a reference
		fcb_get_start_block(fcb) = 0x80000000; // for 0B file
		// zero B file does not occupy any block
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
}

/*
fp: the corresponding FCB index
*/
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
	if (size > (1<<20)) {
		printf("Writing %u B to a file is too large, OS memory error.\n", size);
		assert(0);
	}
	uchar * fcb;
	u32 block_id, block_num;
	fcb = fs_get_fcb(fs, fp);
	// clean up old content
	fs_rm_file_content(fs, fcb); // set bitmap (optional)
	fcb_get_filesize(fcb) = 0;
	// search for a large enough extent, if not, do compact
	block_num = size/fs->STORAGE_BLOCK_SIZE + (size%fs->STORAGE_BLOCK_SIZE > 0);
	block_id = fs_search_freeblock(fs, block_num);
	if (block_id == 0x80000000) {
		// disk compact
		fs_compact(fs);
		block_id = fs_search_freeblock(fs, block_num);
		if (block_id == 0x80000000) {
			printf("Writing %u B to a file is too large, OS memory error.\n", size);
			assert(0);
		}
	}
	// start writing data to file
	u32 pointer = fs->FILE_BASE_ADDRESS + block_id*(fs->STORAGE_BLOCK_SIZE);
	for (int i = 0; i < size; i++) {
		fs->volume[pointer+i] = input[i];
	}
	// setup fcb and super block
	fcb_get_start_block(fcb) = block_id;
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
calling this function.

It will set the bit map for the consecutive 32 blocks (a little overhead)
*/
__device__ void fs_update_size(FileSystem *fs, uchar *fcb, u32 size) {
	u32 block_id, length;
	fcb_get_filesize(fcb) = size;
	block_id = fcb_get_start_block(fcb);
	length = size / fs->STORAGE_BLOCK_SIZE;
	if (size % fs->STORAGE_BLOCK_SIZE) {
		length++;
	}
	// 0B file does not occupy any block
	for (u32 i = 0; i < length; i++) {
		assert(fs_get_superblock(fs, block_id+i) == 0);
		fs_set_superblock(fs, block_id+i, 1);
	}
}

/*
Set the corresponding blocks in super block to be free
*/
__device__ void fs_rm_file_content(FileSystem *fs, uchar *fcb) {
	u32 block_id, size, length;
	block_id = fcb_get_start_block(fcb);
	if (block_id == 0x80000000) {
		// zero B file
		return;
	}
	size = fcb_get_filesize(fcb);
	if (size > 0) {
		length = size / fs->STORAGE_BLOCK_SIZE;
		if (size % fs->STORAGE_BLOCK_SIZE) {
			length++;
		}
	} else {
		length = 0;
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
Return a reference to the bit in the bit map
*/
__device__ int fs_get_superblock(FileSystem *fs, u32 block_id) {
	uchar position = block_id % 8;
	return (fs->volume[block_id/8] >> (7-position)) & 1;
}

/*
Return a copy to the bit in the bit map
*/
__device__ const int fs_get_superblock(const FileSystem *fs, u32 block_id) {
	uchar position = block_id % 8;
	return (fs->volume[block_id/8] >> (7-position)) & 1;
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

__device__ void print_superblock(const FileSystem *fs) {
	// sort fcbs according to start block id
	const uchar **fcbs = new const uchar * [fs->FCB_ENTRIES];
	u32 file_count = 0;
	const uchar * fcb;
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		fcb = fs_get_fcb(fs, i);
		if (fcb_is_valid(fcb)) {
			fcbs[file_count] = fcb;
			file_count++;
		}
	}
	// bubble sort
	bool swap = false;
	for (int i = 0; i < file_count; i++) {
		for (int j = 0; j < file_count-1-i; j++) {
			swap = false;
			if (fcb_get_start_block(fcbs[j]) > fcb_get_start_block(fcbs[j+1])) {
				swap = true;
			}
			if (swap) {
				fcb = fcbs[j];
				fcbs[j] = fcbs[j+1];
				fcbs[j+1] = fcb;
			}
		}
	}
	int i = 0;
	u32 length, block_id, size;
	for (int j = 0; j < 8*fs->SUPERBLOCK_SIZE; j++) {
		if (i >= file_count) {
			if (fs_get_superblock(fs, j) == 1) {
				printf("Block %d is OCCUPIED by no file\n", j);
			}
			continue;
		}
		fcb = fcbs[i];
		size = fcb_get_filesize(fcbs[i]);
		block_id = fcb_get_start_block(fcbs[i]);
		length = size / fs->STORAGE_BLOCK_SIZE;
		length += (size%fs->STORAGE_BLOCK_SIZE) > 0;
		if (fs_get_superblock(fs, j) == 0) {
			if (block_id <= j && j < block_id+length) {
				printf("Block %d is NOT occupied by file %s!\n", j, fcb_get_filename(fcbs[i]));
			}
		} else {
			if (block_id <= j && j < block_id+length) {
				if (j == block_id+length-1) {
					printf("Block %d-%d is occupied by file %s\n", 
						block_id,
						block_id+length-1,
						fcb_get_filename(fcb));
					i++;
				}
				continue;
			}
			printf("Block %d is OCCUPIED by no file\n", j);
		}
	}
	delete fcbs;
}

/*
Search the bitmap to find a free block.
If no free block available, return 0x80000000.

The bitmap is big endian, i.e. higher bits stores blocks in the front.

Since max file size is 1KB = 32 blocks, search free blocks every other 32 blocks
*/
__device__ u32 fs_search_freeblock(const FileSystem *fs, u32 num_blocks) {
	// printf("Requesting %u blocks\n", num_blocks);
	u32 cnt = 0;
	for (u32 i = 0; i < 8*fs->SUPERBLOCK_SIZE; i++) {
		if (fs_get_superblock(fs, i) == 0) {
			cnt++;
			if (cnt == num_blocks) {
				return i-num_blocks+1;
			}
		} else {
			cnt = 0;
		}
	}
	return 0x80000000;
}

__device__ u32 fs_search_freeblock(const FileSystem *fs) {
	return fs_search_freeblock(fs);
}

/*
compact files
*/
__device__ void fs_compact(FileSystem *fs) {
	// sort fcbs according to start block id
	uchar **fcbs = new uchar * [fs->FCB_ENTRIES];
	u32 file_count = 0;
	uchar * fcb;
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		fcb = fs_get_fcb(fs, i);
		if (fcb_is_valid(fcb)) {
			fcbs[file_count] = fcb;
			file_count++;
		}
	}
	// bubble sort
	bool swap = false;
	for (int i = 0; i < file_count; i++) {
		for (int j = 0; j < file_count-1-i; j++) {
			swap = false;
			if (fcb_get_start_block(fcbs[j]) > fcb_get_start_block(fcbs[j+1])) {
				swap = true;
			}
			if (swap) {
				fcb = fcbs[j];
				fcbs[j] = fcbs[j+1];
				fcbs[j+1] = fcb;
			}
		}
	}
	// compact files
	u32 block_id = 0;
	for (int i = 0; i < file_count; i++) {
		fs_move_file_blocks(fs, fcbs[i], block_id);
		block_id += (fcb_get_filesize(fcbs[i])/fs->STORAGE_BLOCK_SIZE + 
			(fcb_get_filesize(fcbs[i])%fs->STORAGE_BLOCK_SIZE > 0));
	}
	delete fcbs;
}

/*
Move blocks of a file to a contiguous set of blocks with starting id
```block_id```. It will raise an error if the blocks to be moved to 
are occupied.
Need to setup fcb and bitmap.
*/
__device__ void fs_move_file_blocks(FileSystem *fs, uchar *fcb, u32 new_block_id) {
	u32 block_id = fcb_get_start_block(fcb);
	if (block_id == new_block_id) {
		return;
	}
	u32 num = fcb_get_filesize(fcb) / fs->STORAGE_BLOCK_SIZE;
	num += fcb_get_filesize(fcb)%fs->STORAGE_BLOCK_SIZE > 0;
	u32 block_pt, new_block_pt;
	block_pt = fs->FILE_BASE_ADDRESS + block_id*(fs->STORAGE_BLOCK_SIZE);
	new_block_pt = fs->FILE_BASE_ADDRESS + new_block_id*(fs->STORAGE_BLOCK_SIZE);
	for (int i = 0; i < num; i++) {
		assert(fs_get_superblock(fs, new_block_id+i) == 0);
		for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++) {
			fs->volume[new_block_pt+j] = fs->volume[block_pt+j];
		}
		block_pt += fs->STORAGE_BLOCK_SIZE;
		new_block_pt += fs->STORAGE_BLOCK_SIZE;
		// set bitmap
		fs_set_superblock(fs, block_id+i, 0);
		fs_set_superblock(fs, new_block_id+i, 1);
	}
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