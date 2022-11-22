#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

__device__ void user_program(FileSystem *fs, uchar *input, uchar *output) {
	/////////////////////// Bonus Test Case ///////////////
	u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input, 64, fp);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_READ);
	fs_read(fs, output, 32, fp);
	fs_gsys(fs, LS_D);
	fs_gsys(fs, LS_S);
	fs_gsys(fs, MKDIR, "app\0");
	fs_gsys(fs, LS_D);
	fs_gsys(fs, LS_S);
	fs_gsys(fs, CD, "app\0");
	fs_gsys(fs, LS_S);
	fp = fs_open(fs, "a.txt\0", G_WRITE);
	fs_write(fs, input + 128, 64, fp);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 256, 32, fp);
	fs_gsys(fs, MKDIR, "soft\0");
	fs_gsys(fs, LS_S);
	fs_gsys(fs, LS_D);
	fs_gsys(fs, CD, "soft\0");
	fs_gsys(fs, PWD);
	fp = fs_open(fs, "A.txt\0", G_WRITE);
	fs_write(fs, input + 256, 64, fp);
	fp = fs_open(fs, "B.txt\0", G_WRITE);
	fs_write(fs, input + 256, 1024, fp);
	fp = fs_open(fs, "C.txt\0", G_WRITE);
	fs_write(fs, input + 256, 1024, fp);
	fp = fs_open(fs, "D.txt\0", G_WRITE);
	fs_write(fs, input + 256, 1024, fp);
	fs_gsys(fs, LS_S);
	fs_gsys(fs, CD_P);
	fs_gsys(fs, LS_S);
	fs_gsys(fs, PWD);
	fs_gsys(fs, CD_P);
	fs_gsys(fs, LS_S);
	fs_gsys(fs, CD, "app\0");
	fs_gsys(fs, RM_RF, "soft\0");
	fs_gsys(fs, LS_S);
	fs_gsys(fs, CD_P);
	fs_gsys(fs, LS_S);

	// /////////////// Test Case 4  ///////////////
    // u32 fp = fs_open(fs, "32-block-0", G_WRITE);
    // fs_write(fs, input, 32, fp);
	// print_superblock(fs);
    // for (int j = 0; j < 1023; ++j) {
    //     char tag[] = "1024-block-????";
    //     int i = j;
    //     tag[11] = static_cast<char>(i / 1000 + '0');
    //     i = i % 1000;
    //     tag[12] = static_cast<char>(i / 100 + '0');
    //     i = i % 100;
    //     tag[13] = static_cast<char>(i / 10 + '0');
    //     i = i % 10;
    //     tag[14] = static_cast<char>(i + '0');
    //     fp = fs_open(fs, tag, G_WRITE);
    //     fs_write(fs, input + j * 1024, 1024, fp);
    // }
    // fs_gsys(fs, RM, "32-block-0");
    // // now it has one 32byte at first, 1023 * 1024 file in the middle

    // fp = fs_open(fs, "1024-block-1023", G_WRITE);
    // printf("triggering gc\n");
    // fs_write(fs, input + 1023 * 1024, 1024, fp);


    // fs_gsys(fs, LS_D);
    // for (int j = 0; j < 1024; ++j) {
    //     char tag[] = "1024-block-????";
    //     int i = j;
    //     tag[11] = static_cast<char>(i / 1000 + '0');
    //     i = i % 1000;
    //     tag[12] = static_cast<char>(i / 100 + '0');
    //     i = i % 100;
    //     tag[13] = static_cast<char>(i / 10 + '0');
    //     i = i % 10;
    //     tag[14] = static_cast<char>(i + '0');
    //     fp = fs_open(fs, tag, G_READ);
    //     fs_read(fs, output + j * 1024, 1024, fp);
    // }

}
