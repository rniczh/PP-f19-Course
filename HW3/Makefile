CC = mpicc
CFLAGS = -Wall -g -O3 -fopenmp
OBJ = conduction.o
TARGET = conduction serial

all: $(OBJ) $(TARGET)

%: %.o
	$(CC) $(CFLAGS) $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test:
	@echo "Not implemented yet."

bench: all
	@echo "MPI version:"
	@time --format=%E mpirun -n 4 ./conduction 20 100000 12345
	@echo "Seiral version:"
	@time --format=%E ./serial 20 100000 12345

record: all
	perf record --call-graph dwarf mpirun -n 4 ./conduction 1000 70000 12345

clean:
	$(RM) $(OBJ) $(TARGET)
