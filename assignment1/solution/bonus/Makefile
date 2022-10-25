CFILES:= $(shell ls|grep .c)
PROGS:=pstree

all: $(PROGS)

%:%.c
	$(CC) -g -o $@ $<

clean:$(PROGS)
	rm $(PROGS)

run:
	make all
	./pstree