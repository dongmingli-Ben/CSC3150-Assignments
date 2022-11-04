.PHONY: source bonus clean

source:
	cd source && \
	nvcc -rdc=true main.cu virtual_memory.cu user_program.cu -o main && \
	./main && \
	cmp data.bin snapshot.bin && \
	cd ..

bonus:
	cd bonus && \
	nvcc -rdc=true main.cu virtual_memory.cu user_program.cu -o main && \
	./main && \
	cmp data.bin snapshot.bin && \
	cd ..

debug:
	cd source && \
	nvcc -g -G -rdc=true main.cu virtual_memory.cu user_program.cu -o main

clean:
	cd source && \
	rm -rf main snapshot.bin && \
	cd .. && \
	cd bonus && \
	rm -rf main snapshot.bin && \
	cd ..