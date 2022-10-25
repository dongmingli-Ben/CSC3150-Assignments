obj-m	:= program2.o
KVERSION := $(shell uname -r)
PWD	:= $(shell pwd)

all:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) modules
clean:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) clean
run:
	gcc test.c -o test
	sudo cp ./test /tmp/test
	sudo insmod program2.ko
	sudo rm /tmp/test
	sleep 5
	sudo rmmod program2.ko
	dmesg | tail -10