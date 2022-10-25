CC=gcc
CFLAGS=-ggdb3 -c -Wall -std=gnu99
LDFLAGS=-pthread
SOURCES=httpserver.c libhttp.c util.c async.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=httpserver

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -g $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) -g $(CFLAGS) $< -o $@

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)
