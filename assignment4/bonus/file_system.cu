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

	fs->CUR_DIR_FCB_INDEX = NULL_FCB_INDEX;
}


/*
Return the FCB index
*/
__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
	uchar * fcb;
	u32 fcb_index;
	fcb_index = fs_search_by_name(fs, s, fs->CUR_DIR_FCB_INDEX);
	if (fcb_index == NULL_FCB_INDEX) {
		// fcb not found, and no free fcb
		printf("Cannot create file %s due to OOM\n", s);
		assert(0);
	}
	fcb = fs_get_fcb(fs, fcb_index);
	if (!fcb_is_valid(fcb)) {
		// not found
		if (op == G_READ) {
			printf("%s does not exist!\n", s);
			assert(0);
		}
		// create file if not exist
		fs_create_pseudo_file(fs, fcb, s, fs->CUR_DIR_FCB_INDEX, 0);
	}
	// print_fcb(fcb);
	return fcb_index;
}

/*
fp: the corresponding FCB index
*/
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
	uchar * fcb;
	u32 filesize;
	fcb = fs_get_fcb(fs, fp);
	filesize = fcb_get_filesize(fcb);
	u32 pointer = fs_get_file_data_index(fs, fcb);
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
	assert(size <= (1 << 10));
	uchar * fcb;
	fcb = fs_get_fcb(fs, fp);
	// clean up old content
	fs_rm_file_content(fs, fcb); // set bitmap
	fcb_get_filesize(fcb) = 0;
	// search for a large enough extent, if not, do compact
	u32 block_id, block_num;
	block_num = size/fs->STORAGE_BLOCK_SIZE + (size%fs->STORAGE_BLOCK_SIZE > 0);
	block_id = fs_search_freeblock(fs, block_num);
	if (block_id == NULL_BLOCK_INDEX) {
		// disk compact
		fs_compact(fs);
		block_id = fs_search_freeblock(fs, block_num);
		if (block_id == NULL_BLOCK_INDEX) {
			printf("Writing %u B to a file is too large, OS memory error.\n", size);
			assert(0);
		}
	}
	// setup fcb and super block
	fcb_get_start_block(fcb) = block_id;
	fs_update_size(fs, fcb, size);
	// start writing data to file
	u32 pointer = fs_get_file_data_index(fs, fcb);
	for (int i = 0; i < size; i++) {
		fs->volume[pointer+i] = input[i];
	}
	// increment time
	gtime++;
	assert((gtime >> 16) == 0);
	fcb_get_modified_time(fcb) = gtime;
	return 0;
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	switch (op)
	{
	case LS_D:
		fs_ls(fs, op);
		break;
	case LS_S:
		fs_ls(fs, op);
		break;
	case CD_P:
		fs_cd_parent_dir(fs);
		break;
	case PWD:
		fs_print_pwd(fs);
		break;
	default:
		printf("%d not recognized\n", op);
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	switch (op)
	{
	case RM:
		fs_rm_file(fs, s);
		break;
	case RM_RF:
		fs_rm_dir(fs, s);
		break;
	case CD:
		fs_cd_child_dir(fs, s);
		break;
	case MKDIR:
		fs_mkdir(fs, s);
		break;
	default:
		printf("%d not recognized\n", op);
		assert(0);
	}
}

/*
linux mkdir equivalent
*/
__device__ void fs_mkdir(FileSystem *fs, char *s) {
	uchar * parent_fcb = fs_get_fcb(fs, fs->CUR_DIR_FCB_INDEX);
	u32 fcb_index = fs_search_by_name(fs, s, fs->CUR_DIR_FCB_INDEX);
	uchar * fcb = fs_get_fcb(fs, fcb_index);
	if (fcb_is_valid(fcb)) {
		printf("Directory %s exists! Cannot create dir.\n", s);
		assert(0);
	}
	fs_create_pseudo_file(fs, fcb, s, fs->CUR_DIR_FCB_INDEX, 1);
}

/*
Create a pseudo file, including file and dir.
It sets up the FCB for the pseudo file, and the parent dir content.
op: 0 ==> create a file, 1 ==> create a dir
*/
__device__ void fs_create_pseudo_file(
		FileSystem *fs, 
		uchar *fcb, 
		char * name, 
		u32 parent_fcb_index, 
		int op) {
	assert(my_strlen(name) < 20);
	my_strcpy(fcb_get_filename(fcb), name);
	fcb_get_filesize(fcb) = 0;  // return a reference
	fcb_get_start_block(fcb) = NULL_BLOCK_INDEX;
	fcb_is_directory(fcb) = op;
	fcb_get_parent_fcb_index(fcb) = parent_fcb_index;
	// increment global time
	gtime++;
	assert((gtime >> 16) == 0);
	// set modified time and created time
	fcb_get_modified_time(fcb) = gtime; // return a reference
	fcb_get_created_time(fcb) = gtime; // return a reference
	// update parent file content
	if (parent_fcb_index != NULL_FCB_INDEX) {
		uchar * parent_fcb = fs_get_fcb(fs, parent_fcb_index);
		fcb_get_filesize(parent_fcb) += my_strlen(name) + 1;
		fcb_get_modified_time(parent_fcb) = gtime; // return a reference
	}
}

/*
Set the CUR_DIR_FCB_INDEX accordingly
*/
__device__ void fs_cd_parent_dir(FileSystem *fs) {
	if (fs->CUR_DIR_FCB_INDEX == NULL_FCB_INDEX) {
		printf("You are at the root directory. Cannot go to parent dir.\n");
		assert(0);
	}
	uchar * fcb;
	fcb = fs_get_fcb(fs, fs->CUR_DIR_FCB_INDEX);
	fs->CUR_DIR_FCB_INDEX = fcb_get_parent_fcb_index(fcb);
}

/*
linux equivalent of cd
*/
__device__ void fs_cd_child_dir(FileSystem *fs, char *s) {
	fs->CUR_DIR_FCB_INDEX = fs_search_by_name(fs, s, fs->CUR_DIR_FCB_INDEX);
}

/*
Print the current directory
*/
__device__ void fs_print_pwd(FileSystem *fs) {
	const uchar * dir_stack[3]; // store fcb entry
	int cur_dir_fcb_index = fs->CUR_DIR_FCB_INDEX;
	int i = 0;
	const uchar * fcb;
	while (cur_dir_fcb_index != NULL_FCB_INDEX) {
		fcb = fs_get_fcb(fs, cur_dir_fcb_index);
		dir_stack[i] = fcb;
		cur_dir_fcb_index = fcb_get_parent_fcb_index(fcb);
		i++;
	}
	i--;
	while (i >= 0) {
		printf("/%s", fcb_get_filename(dir_stack[i]));
		i--;
	}
	printf("\n");
}

/*
Remove a directory
*/
__device__ void fs_rm_dir(FileSystem *fs, char *s) {
	uchar * fcb;
	u32 fcb_index = fs_search_by_name(fs, s, fs->CUR_DIR_FCB_INDEX);
	if (fcb_index == NULL_FCB_INDEX || !fcb_is_valid(fs_get_fcb(fs, fcb_index))) {
		printf("Directory to be removed %s not found at ", s);
		fs_print_pwd(fs);
		assert(0);
	}
	fcb = fs_get_fcb(fs, fcb_index);
	// remove cur dir and sub dirs
	int level = 0;
	u32 dir_stack[3] = {NULL_FCB_INDEX, NULL_FCB_INDEX, NULL_FCB_INDEX}; // for post order traversal, store fcb index
	// starting from the root for the sub-tree to be removed
	dir_stack[level] = fcb_index;
	while (dir_stack[level] != NULL_FCB_INDEX) {
		fcb_index = fs_get_first_child(fs, dir_stack[level]);
		fcb = fs_get_fcb(fs, fcb_index);
		if (fcb_index == NULL_FCB_INDEX || !fcb_is_valid(fcb)) {
			// empty directory
			fs_rm_pseudo_file(fs, fs_get_fcb(fs, dir_stack[level]));
			dir_stack[level] = NULL_FCB_INDEX;
			if (level == 0) break;
			level--;
			continue;
		}
		if (!fcb_is_directory(fcb)) {
			fs_rm_pseudo_file(fs, fcb);
			continue;
		}
		// add dir to stack
		level++;
		dir_stack[level] = fcb_index;
	}
}

/*
Remove a file, cannot remove directories.

Need to update parent dir FCB too.
*/
__device__ void fs_rm_file(FileSystem *fs, char *s) {
	uchar * fcb;
	u32 fcb_index;
	fcb_index = fs_search_by_name(fs, s, fs->CUR_DIR_FCB_INDEX);
	fcb = fs_get_fcb(fs, fcb_index);
	if (fcb_is_directory(fcb)) {
		printf("Cannot remove directory with RM\n");
		assert(0);
	}
	fs_rm_pseudo_file(fs, fcb);
}

/*
Remove a pseudo file, including files and directories.
It clears the file content and update the content of parent dir.

You can remove an empty dir with it.
*/
__device__ void fs_rm_pseudo_file(FileSystem *fs, uchar *fcb) {
	u32 fcb_index;
	uchar * parent_fcb;
	char * name = fcb_get_filename(fcb);
	// set bit map
	fs_rm_file_content(fs, fcb);
	// update parent FCB
	fcb_index = fcb_get_parent_fcb_index(fcb);
	parent_fcb = fs_get_fcb(fs, fcb_index); // parent fcb
	fcb_get_filesize(parent_fcb) -= (my_strlen(name)+1);
	// clean up fcb
	memset(fcb, 0, 32);
}

/*
Return the pointer (index of fs->volume) to the data block
*/
__device__ u32 fs_get_file_data_index(FileSystem *fs, uchar *fcb) {
	u32 block_id = fcb_get_start_block(fcb);
	u32 pointer = fs->FILE_BASE_ADDRESS + block_id*(fs->STORAGE_BLOCK_SIZE);
	return pointer;
}

/*
Return the index of fcb entry of first children of the dir.
If no children is found, return NULL_FCB_INDEX.

It search over the entire FCB entries.
*/
__device__ u32 fs_get_first_child(FileSystem *fs, u32 fcb_index) {
	uchar * fcb;
	u32 child_fcb_index;
	for (child_fcb_index = 0; child_fcb_index < fs->FCB_ENTRIES; child_fcb_index++) {
		fcb = fs_get_fcb(fs, child_fcb_index);
		if (fcb_get_parent_fcb_index(fcb) == fcb_index) {
			return child_fcb_index;
		}
	}
	return NULL_FCB_INDEX;
}

/*
Perform ls
*/
__device__ void fs_ls(const FileSystem * fs, int op) {
	const uchar *fcbs[50];
	const uchar * fcb;
	int file_num = 0;
	// loop over fcb
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		fcb = fs_get_fcb(fs, i);
		if (!fcb_is_valid(fcb) || fcb_get_parent_fcb_index(fcb) != fs->CUR_DIR_FCB_INDEX) {
			continue;
		}
		if (file_num == 50) {
			printf("Directory has more than 50 files and subdir, terminating...\n");
			assert(0);
		}
		fcbs[file_num] = fcb;
		file_num++;
	}
	// sort
	// bubble sort
	const uchar * tmp;
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
		printf("Sorted %d files by modified time:\n", file_num);
	} else {
		printf("Sort %d by file size:\n", file_num);
	}
	for (int i = 0; i < file_num; i++) {
		if (op == LS_D) {
			printf("%s\t%u", fcb_get_filename(fcbs[i]), fcb_get_modified_time(fcbs[i]));
		} else {
			printf("%s\t%u", fcb_get_filename(fcbs[i]), fcb_get_filesize(fcbs[i]));
		}
		if (fcb_is_directory(fcbs[i])) {
			printf("\td\n");
		} else {
			printf("\n");
		}
	}
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
	// assert(block_id % 32 == 0);  // blocks of every file should begin with 32x
	length = size / fs->STORAGE_BLOCK_SIZE;
	if (size % fs->STORAGE_BLOCK_SIZE) {
		length++;
	}
	// memset(&(fs->volume[block_id/8]), 0, 4);
	// fs_set_superblock(fs, block_id, 1);  // the first block should be marked as used even for 0 byte file
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
	if (block_id == NULL_BLOCK_INDEX) {
		// 0B file
		return;
	}
	size = fcb_get_filesize(fcb);
	if (size > 0) {
		length = size / fs->STORAGE_BLOCK_SIZE;
		if (size % fs->STORAGE_BLOCK_SIZE) {
			length++;
		}
	} else {
		// a 0B file should have a null block index
		assert(0);
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
	unsigned short * shorts = (unsigned short *) &(fcb[20]);
	return shorts[2];
}

/*
Return a reference to the created time
*/
__device__ unsigned short & fcb_get_created_time(const uchar * fcb) {
	unsigned short * shorts = (unsigned short *) &(fcb[20]);
	return shorts[3];
}

/*
Return a reference to the start block id
*/
__device__ unsigned short & fcb_get_start_block(const uchar * fcb) {
	unsigned short * shorts = (unsigned short *) &(fcb[20]);
	return shorts[0];
}

/*
Return a reference to the file size
*/
__device__ unsigned short & fcb_get_filesize(const uchar * fcb) {
	unsigned short * shorts = (unsigned short *) &(fcb[20]);
	return shorts[1];
}

/*
Return a reference to the parent FCB index
*/
__device__ unsigned short & fcb_get_parent_fcb_index(const uchar * fcb) {
	unsigned short * shorts = (unsigned short *) &(fcb[20]);
	return shorts[4];
}

/*
Return a reference to the bits that indicate whether 
it is a directory.
1: directory, 0: file
*/
__device__ unsigned short & fcb_is_directory(const uchar * fcb) {
	unsigned short * shorts = (unsigned short *) &(fcb[20]);
	return shorts[5];
}

/*
Get the file name from the FCB
*/
__device__ char * fcb_get_filename(const uchar * fcb) {
	return (char *) fcb;
}

/*
Search for a match entry in FCB.
If no match is found, return a free FCB
*/
__device__ u32 fs_search_by_name(const FileSystem *fs, const char *s, u32 parent_fcb_index) {
	u32 fcb_index = 0;
	u32 free_fcb = NULL_FCB_INDEX;
	const uchar * fcb;
	char * name;
	for (fcb_index = 0; fcb_index < fs->FCB_ENTRIES; fcb_index++) {
		fcb = fs_get_fcb(fs, fcb_index);
		if (!fcb_is_valid(fcb)) {
			free_fcb = fcb_index;
			continue;
		}
		name = fcb_get_filename(fcb);
		if (my_strcmp(name, s) == 0 && fcb_get_parent_fcb_index(fcb) == parent_fcb_index) {
			return fcb_index;
		}
	}
	return free_fcb;
}

/*
Return the bit of the block
*/
__device__ int fs_get_superblock(const FileSystem *fs, u32 block_id) {
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

/*
Search the bitmap to find a free block.
If no free block available, return NULL_BLOCK_INDEX.

The bitmap is big endian, i.e. higher bits stores blocks in the front.
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
	return NULL_BLOCK_INDEX;
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
		if (fcb_is_valid(fcb) & !fcb_is_directory(fcb)) {
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
	// update fcb
	fcb_get_start_block(fcb) = new_block_id;
}

// some debug functions

__device__ void print_fcb(FileSystem *fs, uchar *fcb) {
	printf("--------start of fcb----------\n");
	if (fcb_is_directory(fcb)) {
		printf("dir ");
	} else {
		printf("file ");
	}
	printf("name: %s\n", fcb_get_filename(fcb));
	printf("start block id: %u\n", fcb_get_start_block(fcb));
	printf("size: %u\n", fcb_get_filesize(fcb));
	printf("modified time: %u\n", fcb_get_modified_time(fcb));
	printf("created time: %u\n", fcb_get_created_time(fcb));
	printf("parent fcb index: %u\n", fcb_get_parent_fcb_index(fcb));
	printf("content: ");
	u32 pointer = fs_get_file_data_index(fs, fcb);
	u32 length = my_strlen((char *) &(fs->volume[pointer]));
	int size = fcb_get_filesize(fcb);
	while (size > 0) {
		printf("%s", &fs->volume[pointer]);
		size -= (length + 1);
		pointer += (length + 1);
		length = my_strlen((char *) &(fs->volume[pointer]));
		if (size > 0 && length > 0) {
			printf("\\0");
		}
	}
	printf("\n");
	printf("--------end of fcb----------\n");
}

__device__ void print_superblock(const FileSystem *fs) {
	// sort fcbs according to start block id
	const uchar **fcbs = new const uchar * [fs->FCB_ENTRIES];
	u32 file_count = 0;
	const uchar * fcb;
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		fcb = fs_get_fcb(fs, i);
		if (fcb_is_valid(fcb) && !fcb_is_directory(fcb)) {
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
	printf("------------------super block occupation--------------------\n");
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
	printf("------------------end of super block---------------------------\n");
	delete fcbs;
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

__device__ char *(my_strstr)(const char *s1, const char *s2)
	{	/* find first occurrence of s2[] in s1[] */
	if (*s2 == '\0')
		return ((char *)s1);
	for (; (s1 = my_strchr(s1, *s2)) != NULL; ++s1)
		{	/* match rest of prefix */
		const char *sc1, *sc2;

		for (sc1 = s1, sc2 = s2; ; )
			if (*++sc2 == '\0')
				return ((char *)s1);
			else if (*++sc1 != *sc2)
				break;
		}
	return (NULL);
	}

__device__ char *(my_strchr)(const char *s, int c)
	{	/* find first occurrence of c in char s[] */
	const char ch = c;

	for (; *s != ch; ++s)
		if (*s == '\0')
			return (NULL);
	return ((char *)s);
	}