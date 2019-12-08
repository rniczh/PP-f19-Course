CXX = g++

SRC = $(wildcard *.cpp)
TAR = $(SRC:%.cpp=%)

CFLAGS += -pthread -std=c++11 -O2 -Wno-psabi

.PHONY: all run clean

all: $(TAR)

%: %.cpp
	$(CXX) $(CPPFLAGS) $(CFLAGS) $< -o $@

bench: all
	@echo "========== Original Version ==================="
	@time --format="Time: %E" ./pi 16 1000000000
	@echo "========== Pthread ============================"
	@time --format="Time: %E" ./pi-pthread 16 1000000000
	@echo "========== Pthread + SIMD ====================="
	@time --format="Time: %E" ./pi-pthread-simd 16 1000000000
	@echo "========== Pthread + SIMD + FastRand =========="
	@time --format="Time: %E" ./pi-pthread-simd-fastrand 16 1000000000
	@echo "========== Pthread + SIMD + FastRand(SIMD) ===="
	@time --format="Time: %E" ./pi-pthread-simd-fastrand 16 1000000000

clean:
	$(RM) $(TAR)
