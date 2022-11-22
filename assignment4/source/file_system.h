#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2

struct FileSystem {
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
};


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);

__device__ void print_superblock(const FileSystem *fs);
__device__ void fs_compact(FileSystem *fs);
__device__ void fs_move_file_blocks(FileSystem *fs, uchar *fcb, u32 block_id);
__device__ u32 fs_search_by_name(const FileSystem *fs, const char *s);
__device__ int fs_get_superblock(FileSystem *fs, u32 block_id);
__device__ int fs_get_superblock(const FileSystem *fs, u32 block_id);
__device__ u32 fs_search_freeblock(const FileSystem *fs);
__device__ u32 fs_search_freeblock(const FileSystem *fs, u32 num_blocks);
__device__ void fs_set_superblock(FileSystem *fs, u32 block_id, int op);
__device__ void fs_rm_file_content(FileSystem *fs, uchar *fcb);
__device__ void fs_update_size(FileSystem *fs, uchar *fcb, u32 size);
__device__ const uchar * fs_get_fcb(const FileSystem *fs, u32 index);
__device__ uchar * fs_get_fcb(FileSystem *fs, u32 index);

__device__ char * fcb_get_filename(const uchar * fcb);
__device__ u32 & fcb_get_filesize(const uchar * fcb);
__device__ u32 & fcb_get_start_block(const uchar * fcb);
__device__ unsigned short & fcb_get_modified_time(const uchar * fcb);
__device__ unsigned short & fcb_get_created_time(const uchar * fcb);
__device__ bool fcb_is_valid(const uchar * fcb);


// some standard C library functions

__device__ char *(my_strcpy)(char *s1, const char *s2);
__device__ size_t (my_strlen)(const char *s);
__device__ int (my_strcmp)(const char *s1, const char *s2);

#endif