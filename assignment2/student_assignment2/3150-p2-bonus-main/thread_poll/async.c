
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

// #define DEBUG

#ifdef DEBUG
/* for debug purpose*/
#include <stdio.h>
/* end debug*/
#endif

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
static void (**handler_func)(int);
static int * handler_args;

static tasknode * task_queue_head = NULL;
static threadnode * free_thread_head = NULL;
static pthread_t * threads;

/*mutex lock*/
static pthread_mutex_t task_lock = PTHREAD_MUTEX_INITIALIZER;
/*different condition signal for different threads*/
static pthread_cond_t * cond_arr;

#ifdef DEBUG
/* for debug purpose*/
static void print_task_queue() {
    struct tasknode * node;
    DL_FOREACH(task_queue_head, node) {
        printf("%d->", node->args);
    }
    printf("NULL\n");
}
/*end debug*/
#endif

static void * thread_func(void * args) {
    long id = (long)args;
    void (*func)(int);
    int h_args;
    pthread_mutex_lock(&task_lock);
#ifdef DEBUG
    printf("[thread %ld] locked mutex\n", id);
#endif
    while (1) {
#ifdef DEBUG
        /* for debug purpose */
        print_task_queue();
        /* end debug */
#endif
        if (task_queue_head == NULL) {
            /*wait for jobs*/
#ifdef DEBUG
            printf("[thread %ld] start waiting, release mutex\n", id);
#endif
            pthread_cond_wait(&cond_arr[id], &task_lock);
#ifdef DEBUG
            printf("[thread %ld] acquired mutex from wait\n", id);
#endif
            func = handler_func[id];
            h_args = handler_args[id];
        } else {
            /*put job to global variable*/
            /*job queue is not empty, take the next job*/
            tasknode * node = task_queue_head;
            DL_DELETE(task_queue_head, node);
            func = node->handler;
            h_args = node->args;
            free(node);
#ifdef DEBUG
            printf("[thread %ld] acquairing task %d from queue\n", id, h_args);
#endif
        }
        pthread_mutex_unlock(&task_lock);
#ifdef DEBUG
        printf("[thread %ld] released mutex\n", id);
        printf("[thread %ld] working on %d\n", id, h_args);
#endif
        func(h_args);
#ifdef DEBUG
        printf("[thread %ld] work done on %d\n", id, h_args);
#endif
        /*
        if no job, add worker to free thread queue.
        */
        pthread_mutex_lock(&task_lock);
#ifdef DEBUG
        printf("[thread %ld] locked mutex\n", id);
#endif
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
    handler_func = (void (**) (int)) malloc(sizeof(void (*) (int)) * num_threads);
    handler_args = (int *) malloc(sizeof(int) * num_threads);
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
#ifdef DEBUG
    printf("[async_run] locked mutex\n");
#endif
    if (free_thread_head == NULL) {
        /*all threads are busy*/
        tasknode * node = (tasknode *) malloc(sizeof(tasknode));
        node->handler = handler;
        node->args = args;
        DL_APPEND(task_queue_head, node);
#ifdef DEBUG
        printf("[async_run] added task %d to queue\n", args);
#endif
    } else {
        /*dispatch work to available thread worker*/
        int tid = free_thread_head->tid;
        handler_func[tid] = handler;
        handler_args[tid] = args;
        pthread_cond_signal(&cond_arr[tid]);
        threadnode * node = free_thread_head;
        DL_DELETE(free_thread_head, node);
        free(node);
#ifdef DEBUG
        printf("[async_run] dispatch task %d to thread %d\n", args, tid);
#endif
    }
#ifdef DEBUG
    /* for debug purpose */
    print_task_queue();
    /* end debug */
#endif
    pthread_mutex_unlock(&task_lock);
#ifdef DEBUG
    printf("[async_run] released mutex\n");
#endif
}