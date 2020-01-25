CXX = gcc
CUDAC = nvcc

SRC = $(wildcard *.c)
TAR = $(SRC:%.c=%)

CUSRC = $(wildcard *.cu)
CUTAR = $(SRC:%.cu=%)

CFLAGS += -O2 -lm

.PHONY: all format bench clean

all: $(TAR)

%: %.c
	$(CXX) $(CFLAGS) $< -o $@

%: %.cu
	$(CUDAC) $< -o $@

format:
	clang-format -i *.c
	clang-format -i *.cu

bench: all
	@echo "========== Original Version ==================="
	@time --format="Time: %E" ./serial_wave 10000 10000 > tmp.txt
	@echo "========== Cuda ==============================="
	@time --format="Time: %E" ./wave 10000 10000 > tmp.txt

clean:
	$(RM) $(TAR) tmp.txt
