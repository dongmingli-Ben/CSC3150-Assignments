#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

/* If WIFEXITED(STATUS), the low-order 8 bits of the status.  */
#define __WEXITSTATUS(status) (((status)&0xff00) >> 8)

/* If WIFSIGNALED(STATUS), the terminating signal.  */
#define __WTERMSIG(status) ((status)&0x7f)

/* If WIFSTOPPED(STATUS), the signal that stopped the child.  */
#define __WSTOPSIG(status) __WEXITSTATUS(status)

/* Nonzero if STATUS indicates normal termination.  */
#define __WIFEXITED(status) (__WTERMSIG(status) == 0)

/* Nonzero if STATUS indicates termination by a signal.  */
#define __WIFSIGNALED(status) (((signed char)(((status)&0x7f) + 1) >> 1) > 0)

/* Nonzero if STATUS indicates the child is stopped.  */
#define __WIFSTOPPED(status) (((status)&0xff) == 0x7f)

extern pid_t kernel_clone(struct kernel_clone_args *kargs);
extern int do_execve(struct filename *filename,
		     const char __user *const __user *__argv,
		     const char __user *const __user *__envp);
extern struct filename *getname_kernel(const char *filename);
struct wait_opts {
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;

	struct waitid_info *wo_info;
	int wo_stat;
	struct rusage *wo_rusage;

	wait_queue_entry_t child_wait;
	int notask_error;
};
extern long do_wait(struct wait_opts *wo);

int my_exec(void *x)
{
	printk("[program2] : child process\n");
	printk("[program2] : This is the child process, pid = %d\n",
	       current->pid);
	/* execute a test program in child process */
	struct filename *file;
	file = getname_kernel("/tmp/test");
	int r = do_execve(file, NULL, NULL);
	if (r != 0) {
		printk("child process do_execve failed !!!\n");
		do_exit(r);
	}
	return 0;
}

// implement fork function
int my_fork(void *argc)
{
	// set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	printk("[program2] : This is the parent process, pid = %d\n",
	       current->pid);
	/* fork a process using kernel_clone or kernel_thread */
	struct kernel_clone_args kargs = {
		.exit_signal = SIGCHLD,
		.stack = (unsigned long)my_exec,
		.stack_size = 0,
		.parent_tid = NULL,
		.child_tid = NULL,
		.tls = 0,
	};
	pid_t pid = kernel_clone(&kargs);
	// parent process
	/* wait until child process terminates */
	struct wait_opts wo;
	struct pid *wo_pid = NULL;
	enum pid_type type;
	wo_pid = find_get_pid(pid);

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED | WUNTRACED;
	wo.wo_info = NULL;
	wo.wo_stat = 100;
	wo.wo_rusage = NULL;
	int a;
	a = do_wait(&wo);

	printk("[program2] : get SIGTERM signal\n");
	printk("[program2] : child process terminated\n");

	int status = wo.wo_stat;
	if (__WIFEXITED(status)) {
		printk("[program2] : Normal termination with EXIT STATUS %d\n",
		       __WEXITSTATUS(status));
	} else if (__WIFSTOPPED(status)) {
		printk("[program2] : child process STOPPED with signal %d\n",
		       __WSTOPSIG(status));
	} else if (__WIFSIGNALED(status)) {
		printk("[program2] : child process TERMINATED with signal %d\n",
		       __WTERMSIG(status));
	} else {
		printk("[program2] : child process terminated with unknown "
		       "status %d\n",
		       status);
	}
	put_pid(wo_pid);

	return 0;
}

static int __init program2_init(void)
{
	printk("[program2] : Module_init {%s} {%d}\n", "Li Dongming",
	       119020023);

	/* write your code here */
	printk("[program2] : Module_init create kthread start\n");
	/* create a kernel thread to run my_fork */
	struct task_struct *task;
	task = kthread_create(my_fork, NULL, "my-fork-thread");
	if (!IS_ERR(task)) {
		printk("[program2] : Module_init kthread start\n");
		wake_up_process(task);
	}
	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
