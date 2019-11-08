CXX = g++

SRC = $(wildcard *.cpp)
TAR = $(SRC:%.cpp=%)

CFLAGS += -pthread -std=c++11 -O2

.PHONY: all run clean

all: $(TAR)

%: %.cpp
	$(CXX) $(CPPFLAGS) $(CFLAGS) $< -o $@

bench: all
	perf stat --repeat 5 -e branch-misses,cache-misses,cache-references,instructions,cycles ./pi 16 1000000

clean:
	$(RM) $(TAR)
