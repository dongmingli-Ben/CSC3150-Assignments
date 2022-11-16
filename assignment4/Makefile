.PHONY: source bonus clean

source:
	cd source && \
	nvcc -rdc=true main.cu file_system.cu user_program.cu -o main && \
	./main && \
	cd ..

bonus:
	cd bonus && \
	nvcc -rdc=true main.cu file_system.cu user_program.cu -o main && \
	./main && \
	cd ..

debug:
	cd source && \
	nvcc -g -G -rdc=true main.cu file_system.cu user_program.cu -o main

clean:
	cd source && \
	rm -rf main snapshot.bin && \
	cd .. && \
	cd bonus && \
	rm -rf main snapshot.bin && \
	cd ..