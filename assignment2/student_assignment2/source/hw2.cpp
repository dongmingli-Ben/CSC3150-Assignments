#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
/* for sleep */
#include <errno.h>

#define ROW 10
#define COLUMN 50 


struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 


enum class game_status {
	WIN,
	LOSS,
	PLAYING,
	QUIT,
} status;


pthread_mutex_t map_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t frog_lock = PTHREAD_MUTEX_INITIALIZER;

char map[ROW+10][COLUMN] ; 

/* for debug purpose */
void printf_map(void) {
	for (int i = 0; i <= ROW; i++) {
		for (int j = 0; j < COLUMN; j++) {
			printf("%c", map[i][j]);
		}
		printf("\n");
	}
}
/* end debug */

void printw_map(void) {
	for (int i = 0; i <= ROW; i++) {
		for (int j = 0; j < COLUMN; j++) {
			printw("%c", map[i][j]);
		}
		printw("\n");
	}
}

int ssleep(float t);

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

// inclusive
int randint(int lwr, int upr) {
	int range = upr - lwr + 1;
	return rand() % range + lwr;
}

/* thread for moving logs only (every 0.1 second) */
void *logs_move( void *t ){
	/* log settings */
	srand(time(NULL));
	int front_log_pos[ROW-1];  // the first log in screen that leads other log of the same line
	int log_length[ROW-1];
	int log_space[ROW-1];
	int log_speed[ROW-1];
	char log_direction[ROW-1]; // 'l' or 'r'
	// initialize
	for (int i = 0; i < ROW-1; i++) {
		log_length[i] = randint(10, 20);
		// log_space[i] = randint(5, COLUMN/4);
		log_space[i] = COLUMN - log_length[i];
		// log_speed[i] = randint(1, 2);
		log_speed[i] = 1;
		log_direction[i] = i%2 == 0 ? 'l' : 'r';
		front_log_pos[i] = log_direction[i] == 'l' ? randint(0, log_space[i]) : 
							randint(COLUMN-log_space[i]-1, COLUMN-1);
	}

	/*  Move the logs  */
	while (status == game_status::PLAYING) {
		// update log status
		for (int i = 0; i < ROW-1; i++) {
			if (log_direction[i] == 'l') {
				front_log_pos[i] -= log_speed[i];
				if (front_log_pos[i]+log_length[i] <= 0) {
					front_log_pos[i] += log_length[i] + log_space[i];
				}
			} else {
				front_log_pos[i] += log_speed[i];
				if (front_log_pos[i]-log_length[i] >= COLUMN-1) {
					front_log_pos[i] -= (log_length[i] + log_space[i]);
				}
			}
		}
		pthread_mutex_lock(&map_lock);
		memset(map, 0, sizeof(map));
		for(int i = 1; i < ROW; ++i ){	
			for(int j = 0; j < COLUMN - 1; ++j )	
				map[i][j] = ' ' ;  
		}	
		memset(map[0], '|', COLUMN*sizeof(char));
		memset(map[ROW], '|', COLUMN*sizeof(char));
		for (int i = 0; i < ROW-1; i++) {
			int row = i+1;
			int pos = front_log_pos[i];
			int d = log_direction[i] == 'l' ? 1 : -1;
			int length = log_length[i] * d;
			int space = log_space[i] * d;
			int end = pos + length - d;
			while ((0 <= pos && pos < COLUMN) || (0 <= end && end < COLUMN)) {
				int a = pos < end ? pos : end;
				int b = pos < end ? end : pos;
				a = a >= 0 ? a : 0;
				b = b < COLUMN ? b: COLUMN-1;
				memset((map[row]+a), '=', (b-a+1) * sizeof(char));
				pos += length + space;
				end = pos + length - d;
			}
		}
		pthread_mutex_unlock(&map_lock);
		// draw logs
		// update frog position
		pthread_mutex_lock(&frog_lock);
		if ((frog.y > 0 && frog.y < ROW)) {
			int row = frog.y-1;
			if (log_direction[row] == 'l') {
				frog.x -= log_speed[row];
			} else {
				frog.x += log_speed[row];
			}
			if ((frog.x < 0 || frog.x >= COLUMN)) {
				pthread_mutex_unlock(&frog_lock);
				pthread_mutex_lock(&map_lock);
				status = game_status::LOSS;
				pthread_mutex_unlock(&map_lock);
				break;
			}
		}
		pthread_mutex_lock(&map_lock);
		map[frog.y][frog.x] = '0';
		pthread_mutex_unlock(&map_lock);
		pthread_mutex_unlock(&frog_lock);
		ssleep(0.1);
	}

	/*  Check keyboard hits, to change frog's position or quit the game. */

	
	/*  Check game's status  */


	/*  Print the map on the screen  */

	
}

/* listen to keyboard hit and act correspondingly*/
void * player_hit(void * t) {
	/* listen to keyboard every 0.05 s */
	while (status == game_status::PLAYING) {
		if (kbhit()) {
			char k = getchar();
			pthread_mutex_lock(&frog_lock);
			if (k == 'w') {
				frog.y -= 1;
			} else if (k == 's' && frog.y < ROW) {
				frog.y += 1;
			} else if (k == 'a' && frog.x > 0) {
				frog.x -= 1;
			} else if (k == 'd' && frog.x < COLUMN-1) {
				frog.x += 1;
			} else if (k == 'q') {
				pthread_mutex_lock(&map_lock);
				status = game_status::QUIT;
				pthread_mutex_unlock(&map_lock);
				pthread_mutex_unlock(&frog_lock);
				break;
			}
			if (map[frog.y][frog.x] == ' ') {
				status = game_status::LOSS;
				pthread_mutex_unlock(&frog_lock);
				break;
			}
			if (frog.y == 0) {
				pthread_mutex_lock(&map_lock);
				status = game_status::WIN;
				pthread_mutex_unlock(&map_lock);
				pthread_mutex_unlock(&frog_lock);
				break;
			}
			pthread_mutex_unlock(&frog_lock);
		}
		ssleep(0.05);
	}
}

int main( int argc, char *argv[] ){

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( (COLUMN-1) / 2, ROW ) ; 
	map[frog.y][frog.x] = '0' ; 

	//Print the map into screen
	// for( i = 0; i <= ROW; ++i)	
	// 	puts( map[i] );

	/* initialize game status */
	status = game_status::PLAYING;
	game_status st = status;

	/*  Create pthreads for wood move and frog control.  */
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	pthread_t t_log, t_player;
	int rc_log = pthread_create(&t_log, &attr, logs_move, NULL);
	int rc_player = pthread_create(&t_player, &attr, player_hit, NULL);
	if (rc_log || rc_player) {
		perror("thread creation failure\n");
		exit(EXIT_FAILURE);
	}
	// pthread_join(t_log, NULL);
	/* 0.1s per frame */

	initscr();
	while (st == game_status::PLAYING) {
		/* refresh the screen */
		erase();
		/* display new frame */
		// for (i = 0; i <= ROW; i++) {
		// 	printw("%s\n", map[i]);
		// }
		pthread_mutex_lock(&map_lock);
		printw_map();
		st = status;
		pthread_mutex_unlock(&map_lock);
		refresh();
		/* sleep */
		ssleep(0.01);
	}
	
	/*  Display the output for user: win, lose or quit.  */
	/* end win */
	endwin();
	printf("\033[2J\033[H");
	switch (status)
	{
	case game_status::WIN:
		printf("You win the game!\n");
		break;
	case game_status::LOSS:
		printf("You lose the game!\n");
		break;
	case game_status::QUIT:
		printf("You quit the game!\n");
		break;
	}
	return 0;

}


int ssleep(float t) {
	struct timespec ts;
	int res;
	ts.tv_sec = (long) t;
	ts.tv_nsec = (t - ts.tv_sec) * 1000000000;

	do {
		res = nanosleep(&ts, &ts);
	} while (res && errno == EINTR);

	return res;
}