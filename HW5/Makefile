CXX=g++
CXXFLAGS=-O2 -DDEBUG
LDFLAGS=-lOpenCL
TEST=test.bmp

SRC = histogram.cpp
TAR = histogram

all: $(TAR)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

bench: all
	./$(TAR) $(TEST)

test: bench
	diff original_$(TEST) hist_$(TEST)

clean:
	$(RM) hist_$(TEST) $(TAR)
