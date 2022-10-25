#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct PTreeNode {
	int pid;
	char comm[256];
	char mesg[256];
	struct PTreeNode *parent;
	// structure child process as linked list
	struct PTreeNode *next_sibling;
	struct PTreeNode *first_child;
	struct PTreeNode *last_child;
};

int filter_proc_files(const struct dirent *d)
{
	int is_proc = 0;
	for (int i = 0; i < 256; i++) {
		if (d->d_name[i] == 0) {
			return is_proc;
		}
		if ('0' <= d->d_name[i] && d->d_name[i] <= '9') {
			is_proc = 1;
		} else {
			return 0;
		}
	}
}

int sort_by_name(const struct dirent **file1, const struct dirent **file2)
{
	char *ptr;
	return strtol((*file1)->d_name, &ptr, 10) -
	       strtol((*file2)->d_name, &ptr, 10);
}

int get_proc_files(struct dirent ***restrict namelist)
{
	int n;
	n = scandir("/proc", namelist, filter_proc_files, sort_by_name);
	if (n == -1) {
		perror("scandir");
		exit(EXIT_FAILURE);
	}
	return n;
}

struct print_tree_args {
	int print_pid;
	int print_pgid;
	int ascii;
	int sort_name;
};

struct PTreeNode *build_process_tree(struct dirent **restrict namelist, int n,
				     struct print_tree_args *args);

void print_tree(struct PTreeNode *root, struct print_tree_args *args);

void free_tree(struct PTreeNode *root);

void sort_children_by_name(struct PTreeNode *root);

int main(int argc, char *argv[])
{
	struct print_tree_args args = {
		.print_pid = 0,
		.print_pgid = 0,
		.ascii = 0,
		.sort_name = 1,
	};
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-p") == 0) {
			args.print_pid = 1;
		}
		if (strcmp(argv[i], "-g") == 0) {
			args.print_pgid = 1;
		}
		if (strcmp(argv[i], "-A") == 0) {
			args.ascii = 1;
		}
		if (strcmp(argv[i], "-n") == 0) {
			args.sort_name = 0;
		}
		if (strcmp(argv[i], "-V") == 0) {
			// show version info
			printf("pstree (dongmingli-Ben)\n");
			printf("This is a free software developed for csc3150 and comes with NO WARRANTY.\n");
			exit(EXIT_SUCCESS);
		}
	}
	struct dirent **namelist;
	struct PTreeNode *process_tree;
	int n;
	n = get_proc_files(&namelist);
	process_tree = build_process_tree(namelist, n, &args);
	print_tree(process_tree, &args);
	free_tree(process_tree);
	// while (n--) {
	//     printf("%s\n", namelist[n]->d_name);
	//     free(namelist[n]);
	// }
	// free(namelist);
	exit(EXIT_SUCCESS);
}

char *whitespace_str(int n)
{
	char *string;
	string = (char *)malloc(sizeof(char) * (n + 1));
	for (int i = 0; i < n; i++) {
		string[i] = ' ';
	}
	string[n] = '\0';
	return string;
}

int children_num(struct PTreeNode *root)
{
	struct PTreeNode *node;
	int len = 0;
	node = root->first_child;
	while (node != NULL) {
		len++;
		node = node->next_sibling;
	}
	return len;
}

void sort_children_by_name(struct PTreeNode *root)
{
	if (root == NULL) {
		return;
	}
	if (root->first_child == NULL) {
		return;
	}
	if (root->first_child == root->last_child) {
		return;
	}
	// bubble sort
	struct PTreeNode *cur_node, *next_node, *pre_node;
	int n = 1;
	for (int i = 0; i < children_num(root); i++) {
		cur_node = root->first_child;
		while (cur_node->next_sibling != NULL) {
			next_node = cur_node->next_sibling;
			if (strcmp(cur_node->comm, next_node->comm) > 0) {
				struct PTreeNode *tmp;
				tmp = next_node->next_sibling;
				if (cur_node == root->first_child) {
					// head
					root->first_child = next_node;
					next_node->next_sibling = cur_node;
					cur_node->next_sibling = tmp;
				} else {
					// non head
					pre_node->next_sibling = next_node;
					next_node->next_sibling = cur_node;
					cur_node->next_sibling = tmp;
				}
				pre_node = next_node;
			} else {
				pre_node = cur_node;
				cur_node = next_node;
			}
		}
		root->last_child = cur_node;
	}
}

void print_tree_with_prefix(struct PTreeNode *root, const char *prefix,
			    struct print_tree_args *args)
{
	printf("%s", root->mesg);
	if (root->first_child == NULL) {
		// no child
		printf("\n");
		return;
	}
	char new_prefix[256];
	strcpy(new_prefix, prefix);
	char *whitespace;
	whitespace = whitespace_str(strlen(root->mesg));
	strcat(new_prefix, whitespace);
	free(whitespace);
	if (root->first_child == root->last_child) {
		// only one child
		if (args->ascii) {
			printf("---");
		} else {
			printf("───");
		}
		strcat(new_prefix, "   ");
		print_tree_with_prefix(root->first_child, new_prefix, args);
	} else {
		// more than one children
		if (args->sort_name) {
			sort_children_by_name(root);
		}
		char root_prefix[256];
		strcpy(root_prefix, new_prefix);
		if (args->ascii) {
			printf("-+-");
			strcat(new_prefix, " | ");
		} else {
			printf("─┬─");
			strcat(new_prefix, " │ ");
		}
		// first child (no need to print prefix, already printed)
		struct PTreeNode *node;
		node = root->first_child;
		print_tree_with_prefix(node, new_prefix, args);
		// other children
		node = node->next_sibling;
		while (node != root->last_child) {
			// except last sibling
			printf("%s", root_prefix);
			if (args->ascii) {
				printf(" |-");
			} else {
				printf(" ├─");
			}
			print_tree_with_prefix(node, new_prefix, args);
			node = node->next_sibling;
		}
		// last child
		printf("%s", root_prefix);
		if (args->ascii) {
			printf(" `-");
		} else {
			printf(" └─");
		}
		print_tree_with_prefix(node, new_prefix, args);
	}
}

void print_tree(struct PTreeNode *root, struct print_tree_args *args)
{
	if (root == NULL) {
		perror("tree is empty");
		exit(EXIT_FAILURE);
	}
	print_tree_with_prefix(root, "", args);
}

struct PTreeNode *build_process_tree(struct dirent **restrict namelist, int n,
				     struct print_tree_args *args)
{
	int max_pid;
	max_pid = 2000;
	struct PTreeNode **process_array, *pnode;
	process_array = (struct PTreeNode **)malloc(sizeof(struct PTreeNode *) *
						    max_pid);
	for (int i = 0; i < n; i++) {
		// loop over processes
		char path[256] = "/proc/";
		// printf("%s\n", namelist[i]->d_name);
		strcat(path, namelist[i]->d_name);
		strcat(path, "/stat");
		// open file and read
		FILE *file = fopen(path, "r");
		if (file == NULL) {
			perror("file not open successfully");
		}
		int pid, ppid, pgid;
		char comm[256];
		char state;
		int result;
		// result = fscanf(file, "%d\t(%s)\t%c %d %d", &pid, comm, &state, &ppid, &pgid);
		// if (result != 5) {
		//     perror("read not as expected");
		//     exit(EXIT_FAILURE);
		// }
		fscanf(file, "%d", &pid);
		fscanf(file, " (%s)", comm);
		fscanf(file, " %c", &state);
		fscanf(file, " %d", &ppid);
		fscanf(file, " %d", &pgid);
		fclose(file);
		char *pt = strrchr(comm, ')');
		if (pt != NULL) {
			*pt = '\0';
		}
		pnode = (struct PTreeNode *)malloc(sizeof(struct PTreeNode));
		pnode->pid = pid;
		strcpy(pnode->comm, comm);
		pnode->next_sibling = NULL;
		pnode->first_child = NULL;
		pnode->last_child = NULL;
		strcpy(pnode->mesg, pnode->comm);
		char str[6];
		if (args->print_pid && args->print_pgid) {
			strcat(pnode->mesg, "(");
			sprintf(str, "%d", pid);
			strcat(pnode->mesg, str);
			strcat(pnode->mesg, ",");
			sprintf(str, "%d", pgid);
			strcat(pnode->mesg, str);
			strcat(pnode->mesg, ")");
		} else if (args->print_pid) {
			strcat(pnode->mesg, "(");
			sprintf(str, "%d", pid);
			strcat(pnode->mesg, str);
			strcat(pnode->mesg, ")");
		} else if (args->print_pgid) {
			strcat(pnode->mesg, "(");
			sprintf(str, "%d", pgid);
			strcat(pnode->mesg, str);
			strcat(pnode->mesg, ")");
		}
		if (ppid == 0) {
			pnode->parent = NULL;
		} else {
			pnode->parent = process_array[ppid];
			struct PTreeNode *last_sibling;
			last_sibling = process_array[ppid]->last_child;
			if (last_sibling == NULL) {
				process_array[ppid]->first_child = pnode;
				process_array[ppid]->last_child = pnode;
			} else {
				process_array[ppid]->last_child->next_sibling =
					pnode;
				process_array[ppid]->last_child = pnode;
			}
		}
		if (pid >= max_pid) {
			// enlarge the array
			max_pid = pid + 500;
			process_array = (struct PTreeNode **)realloc(
				process_array,
				sizeof(struct PTreeNode *) * max_pid);
			if (process_array == NULL) {
				perror("Enlarge array failure");
				exit(EXIT_FAILURE);
			}
		}
		process_array[pid] = pnode;
	}
	// free nodes that are not children of 1
	for (int i = 2; i < max_pid; i++) {
		if (process_array[i]->parent != NULL) {
			break;
		}
		free_tree(process_array[i]);
	}
	struct PTreeNode *root;
	root = process_array[1];
	free(process_array);
	return root;
}

void free_tree(struct PTreeNode *root)
{
	if (root != NULL) {
		// free the children first
		struct PTreeNode *node, *next_node;
		node = root->first_child;
		while (node != NULL) {
			next_node = node->next_sibling;
			free_tree(node);
			node = next_node;
		}
		// free the root
		free(root);
	}
}