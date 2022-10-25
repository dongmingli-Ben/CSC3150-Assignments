#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

extern pid_t kernel_clone(struct kernel_clone_args *kargs);
extern int do_execve(struct filename *filename,
	const char __user *const __user *__argv,
	const char __user *const __user *__envp);
extern struct filename * getname(const char __user * filename);
struct wait_opts {
	enum pid_type		wo_type;
	int			wo_flags;
	struct pid		*wo_pid;

	struct waitid_info	*wo_info;
	int			wo_stat;
	struct rusage		*wo_rusage;

	wait_queue_entry_t		child_wait;
	int			notask_error;
};
extern long do_wait(struct wait_opts *wo);

//implement fork function
int my_fork(void *argc){
	
	
	//set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	/* fork a process using kernel_clone or kernel_thread */
	struct kernel_clone_args kargs = {
		.exit_signal = SIGCHLD,
		.stack = &kthreadd,
		.stack_size = 0
	};
	pid_t pid = kernel_clone(&kargs);
	if (pid == 0) {
		// child process
		printk("[program2] : child process\n");
		/* execute a test program in child process */
		do_execve(getname("/home/vagrant/assignment1/source/program2/test"), NULL, NULL);
	} else {
		// parent process
		/* wait until child process terminates */
		int status;
		struct wait_opts wo;
		struct pid * wo_pid = NULL;
		enum pid_type type;
		wo_pid = find_get_pid(pid);

		wo.wo_type = type;
		wo.wo_pid = wo_pid;
		wo.wo_flags = WEXITED;
		wo.wo_info = NULL;
		wo.wo_stat = status;
		wo.wo_rusage = NULL;

		int a;
		a = do_wait(&wo);

		printk("[program2] : get SIGTERM signal\n");
		printk("[program2] : child process terminated\n");
		printk("[program2] : The return signal is %d\n", wo.wo_stat);
	}
	
	return 0;
}

static int __init program2_init(void){

	printk("[program2] : Module_init {%s} {%d}\n", "Li Dongming", 119020023);
	
	/* write your code here */
	printk("[program2] : Module_init create kthread start\n");
	/* create a kernel thread to run my_fork */
	struct task_struct * task;
	task = kthread_create(&my_fork, NULL, "my-fork-thread");
	if (!IS_ERR(task)) {
		printk("[program2] : Module_init kthread start\n");
		printk("The child process has pid = %d\n", task->pid);
		printk("This is the parent process, pid = %d\n", current->pid);
		wake_up_process(task);
	}
	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
