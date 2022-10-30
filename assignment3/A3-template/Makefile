all:
	nvcc -rdc=true main.cu virtual_memory.cu user_program.cu -o main

debug:
	nvcc -g -G -rdc=true main.cu virtual_memory.cu user_program.cu -o main

clean:
	rm -rf main snapshot.bin