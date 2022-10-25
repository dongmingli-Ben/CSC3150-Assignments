
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

typedef struct threadnode {
    int tid;
    struct threadnode *prev; /* needed for a doubly-linked list only */
    struct threadnode *next; /* needed for singly- or doubly-linked lists */
} threadnode;

typedef struct tasknode {
    int args;
    void (*handler)(int);
    struct tasknode *prev; /* needed for a doubly-linked list only */
    struct tasknode *next; /* needed for singly- or doubly-linked lists */
} tasknode;

/*global variable for async_run*/
static void (*handler_func)(int);
static int handler_args;

static tasknode * task_queue_head = NULL;
static threadnode * free_thread_head = NULL;
static pthread_t * threads;

/*mutex lock*/
static pthread_mutex_t task_lock = PTHREAD_MUTEX_INITIALIZER;
/*different condition signal for different threads*/
static pthread_cond_t * cond_arr;

static void * thread_func(void * args) {
    long id = (long)args;
    void (*func)(int);
    int h_args;
    pthread_mutex_lock(&task_lock);
    while (1) {
        if (task_queue_head == NULL) {
            /*wait for jobs*/
            pthread_cond_wait(&cond_arr[id], &task_lock);
        } else {
            /*put job to global variable*/
            /*job queue is not empty, take the next job*/
            tasknode * node = task_queue_head;
            DL_DELETE(task_queue_head, node);
            handler_func = node->handler;
            handler_args = node->args;
            free(node);
        }
        func = handler_func;
        h_args = handler_args;
        pthread_mutex_unlock(&task_lock);
        func(h_args);
        /*
        if no job, add worker to free thread queue.
        */
        pthread_mutex_lock(&task_lock);
        if (task_queue_head != NULL) {
            continue;
        }
        /*no job, add the thread to free workers list*/
        threadnode * node = (threadnode *) malloc(sizeof(threadnode));
        node->tid = id;
        DL_APPEND(free_thread_head, node);
    }
    return NULL;
}

void async_init(int num_threads) {
    /** TODO: create num_threads threads and initialize the thread pool **/
    threads = (pthread_t *) malloc(sizeof(pthread_t) * num_threads);
    cond_arr = (pthread_cond_t *) malloc(sizeof(pthread_cond_t) * num_threads);
    /*initialize the threads*/
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for (int i = 0; i < num_threads; i++) {
        int rc = pthread_create(&threads[i], &attr, thread_func, (void *)(long)i);
        // pthread_cond_init(&cond_arr[i], NULL);
        if (rc) {
            /*fail to create thread*/
            exit(EXIT_FAILURE);
        }
        /*add thread to free threads*/
        threadnode * node = (threadnode *) malloc(sizeof(threadnode));
        node->tid = i;
        DL_APPEND(free_thread_head, node);
    }
    return;
}

void async_run(void (*handler)(int), int args) {
    // hanlder(args);
    /** TODO: rewrite it to support thread pool **/
    pthread_mutex_lock(&task_lock);
    if (free_thread_head == NULL) {
        /*all threads are busy*/
        tasknode * node = (tasknode *) malloc(sizeof(tasknode));
        node->handler = handler;
        node->args = args;
        DL_APPEND(task_queue_head, node);
    } else {
        /*dispatch work to available thread worker*/
        int tid = free_thread_head->tid;
        handler_func = handler;
        handler_args = args;
        pthread_cond_signal(&cond_arr[tid]);
        threadnode * node = free_thread_head;
        DL_DELETE(free_thread_head, node);
        free(node);
    }
    pthread_mutex_unlock(&task_lock);
}