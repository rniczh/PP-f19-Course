CXX = gcc

SRC = $(wildcard *.c)
TAR = $(SRC:%.c=%)

CFLAGS += -O2 -lm

.PHONY: all run clean

all: $(TAR)

%: %.c
	$(CXX) $(CFLAGS) $< -o $@

format:
	clang-format -i *.c

bench: all
	@echo "========== Original Version ==================="
	@time --format="Time: %E" ./serial_wave 1000 1000
# @echo "========== Cuda ==============================="
# @time --format="Time: %E" ./pi-pthread 16 1000000000

clean:
	$(RM) $(TAR)
