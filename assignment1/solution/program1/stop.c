#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
	printf("------------CHILD PROCESS START------------\n");
	printf("This is the SIGSTOP program\n\n");
	raise(SIGSTOP);
	sleep(5);
	// raise(SIGCONT);
	printf("------------CHILD PROCESS END------------\n");

	return 0;
}
