CC=g++
LDFLAGS=-pthread -lncurses
SOURCES=hw2.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=hw2

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

test: test.o
	$(CC) test.o -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) -c -std=c++11 $< -o $@

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)
