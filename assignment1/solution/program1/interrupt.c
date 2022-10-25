#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
	printf("------------CHILD PROCESS START------------\n");
	printf("This is the SIGINT program\n\n");
	raise(SIGINT);
	sleep(5);
	printf("------------CHILD PROCESS END------------\n");

	return 0;
}
