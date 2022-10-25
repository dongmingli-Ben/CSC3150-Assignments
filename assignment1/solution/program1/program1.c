#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
	/* fork a child process */
	printf("I am the Parent Process, my pid = %d\n", getpid());

	pid_t pid;
	int status;
	pid = fork();
	/* execute test program */
	if (pid < 0) {
		printf("Child process not created successfuly");
	} else if (pid == 0) {
		// child process
		printf("I am the Child Process, my pid = %d\n", getpid());
		char *child_args[argc];
		for (int i = 1; i < argc; i++) {
			child_args[i - 1] = argv[i];
		}
		child_args[argc - 1] = NULL;
		// execute
		execv(child_args[0], child_args);
	} else {
		/* wait for child process terminates */
		waitpid(pid, &status, WUNTRACED);
		/* check child process'  termination status */
		printf("Parent process receives SIGCHLD signal\n");
		if (WIFEXITED(status)) {
			printf("Normal termination with EXIT STATUS %d\n",
			       WEXITSTATUS(status));
		} else if (WIFSTOPPED(status)) {
			printf("CHILD PROCESS STOPPED\n");
		} else if (WIFSIGNALED(status)) {
			printf("CHILD PROCESS TERMINATED with signal \"%s\"\n",
			       strsignal(WTERMSIG(status)));
		}
	}
	return 0;
}
