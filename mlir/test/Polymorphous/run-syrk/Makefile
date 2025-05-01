CC=gcc-13
# CC=/home/f00632390/.local/bin/clang++
# CC=/home/f00632390/llvm-project/build-bisheng/bin/clang++

# To debug using small dataset: make debug
# To test using large dataset: make profile (or just make)
# To debug with different dataset: make debug DEBUG_DATASET=MINI_DATASET/LARGE_DATASET/etc.

CFLAGS=-O3 
POLYBENCH_FLAGS=-DPOLYBENCH_TIME -DDATA_TYPE_IS_FLOAT ../utilities/polybench.c
DEBUG_DATASET=SMALL_DATASET
DEBUG_FLAGS=-DPOLYBENCH_DUMP_ARRAYS -D$(DEBUG_DATASET)
PROFILE_FLAGS=-DLARGE_DATASET
OPENMP=-fopenmp
POLYBENCH_HEADER=../utilities/

NUM_THREADS=16

BENCH_NAME=syrk

TARGETS=profile

PLUTO_PARALLEL_DIMENSIONS=-1

# Clang specific things 
ifneq ($(CC),gcc-13)
OPENMP+=-I/usr/lib/gcc/aarch64-linux-gnu/9/include/ -lomp
endif

# pluto parallel flag for validator
ifneq ($(PLUTO_PARALLEL_DIMENSIONS),-1)
PLUTO_PARALLEL=-parallel-dimensions=$(PLUTO_PARALLEL_DIMENSIONS)
endif

all: $(TARGETS) 

$(BENCH_NAME)_mlir_out.c: $(BENCH_NAME).mlir 
	mlir-affine-validator --polybench-codegen $(PLUTO_PARALLEL) $^ -output x.mlir --codegen-output $@

codegen-debug:
	mlir-opt -correct-schedule -affine-to-scop -affine-cloog-generate  $(BENCH_NAME).mlir -o x.mlir

codegen: $(BENCH_NAME)_mlir_out.c

debug: $(BENCH_NAME).c $(BENCH_NAME)_mlir_out.c
	$(CC) $(CFLAGS) $(POLYBENCH_FLAGS) $(DEBUG_FLAGS) $(OPENMP) -I$(POLYBENCH_HEADER) $(BENCH_NAME).c -DNAIVE_VERSION -o $(BENCH_NAME)-naive -lm
	$(CC) $(CFLAGS) $(POLYBENCH_FLAGS) $(DEBUG_FLAGS) $(OPENMP) -I$(POLYBENCH_HEADER) $(BENCH_NAME).c $(BENCH_NAME)_mlir_out.c -o $(BENCH_NAME)-opt -lm
	./$(BENCH_NAME)-naive 2> naive.log
	OMP_NUM_THREADS=$(NUM_THREADS) OMP_PROC_BIND=true ./$(BENCH_NAME)-opt 2> opt.log
	diff naive.log opt.log

profile: $(BENCH_NAME).c $(BENCH_NAME)_mlir_out.c
	$(CC) $(CFLAGS) $(POLYBENCH_FLAGS) $(PROFILE_FLAGS) $(OPENMP) -I$(POLYBENCH_HEADER) $(BENCH_NAME).c -DNAIVE_VERSION -o $(BENCH_NAME)-naive -lm
	$(CC) $(CFLAGS) $(POLYBENCH_FLAGS) $(PROFILE_FLAGS) $(OPENMP) -I$(POLYBENCH_HEADER) $(BENCH_NAME).c $(BENCH_NAME)_mlir_out.c -o $(BENCH_NAME)-opt -lm
	# ./$(BENCH_NAME)-naive
	OMP_NUM_THREADS=$(NUM_THREADS) OMP_PROC_BIND=true ./$(BENCH_NAME)-opt

pydsl_CTarget:
	python3 $(BENCH_NAME)-affine-transform.py CTARGET SMALL_DATASET CTEST




pydsl_PolyCTarget:
	OMP_NUM_THREADS=$(NUM_THREADS) OMP_PROC_BIND=true  python3 $(BENCH_NAME)-affine-transform.py POLYCTARGET SMALL_DATASET CTEST




clean:
	rm -f $(BENCH_NAME)-naive $(BENCH_NAME)-opt *.log *.out $(BENCH_NAME)_mlir_out.c x.mlir