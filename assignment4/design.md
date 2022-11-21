# Assignment 4 Design

## Demand & Specification

Meta data:

- modified time (for linux, the modified time for a directory is when a file is created or deleted under the directory)
- created time
- file size
- file name

Volume sizes:

- super block: 4KB = 32K bit
- files content: 1024KB
- max files: 1024
- max size of files: 1KB
- block size: 32B
- num of blocks: 32KB

- file control block size: 32KB
- FCB block num: 1KB

FCB content (contiguous allocation): 32B

- file name 20B
- start block: >= 15 bits; main: 4B, bonus: 2B
- file size -> num blocks: >= 10 bits; main: 4B, bonus: 2B
- modified time: counter/timestamp, counter: >= 10 bits; main: 2B
- created time: counter/timestamp, counter: >= 10 bits; main: 2B
- bonus tree directory: index to parent dir: >= 10 bits, bonus: 2B
- directory indicator: 1 bit; bonus: 2B

**Note**: The file content for a directory is always null.

**Update**: need to do file compact when there is external fragmentation.