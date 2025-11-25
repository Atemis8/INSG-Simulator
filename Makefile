# === Build Configuration ===
MODE ?= opt

HAS_MPI := $(shell which mpicc > /dev/null 2>&1 && echo 1 || echo 0)

MPI_DIR := /usr/local
CFLAGS_MPI := -I$(MPI_DIR)/include
LDFLAGS_MPI := -L$(MPI_DIR)/lib -lmpi

LLVM_DIR := /usr/local/opt/llvm
CFLAGS_OMP :=  -I$(LLVM_DIR)/include -DUSE_MPI # -fopenmp # -DUSE_OPENMP
LDFLAGS_OMP := -L$(LLVM_DIR)/lib -lomp
#CC := $(LLVM_DIR)/bin/clang

CC := mpicc

ifeq ($(MODE), debug)
    CFLAGS_MODE := -O0 -fsanitize=address,leak -Wall -g 
    LDFLAGS_MODE := -fsanitize=address,leak -g 
else ifeq ($(MODE), opt)
    CFLAGS_MODE := -O3 -ffast-math -march=native -flto
    LDFLAGS_MODE := -flto
else
    $(error "Unknown mode: $(MODE). Use 'debug' or 'opt'.")
endif

# === Project Structure ===
SRC_DIR := src
INC_DIR := include
OBJ_DIR := obj
EXEC := main

# === External Dependencies ===
PETSC_DIR := /usr/local/petsc
PETSC_INC := $(PETSC_DIR)/include
PETSC_LIB := $(PETSC_DIR)/lib
PETSC_LIBNAME := -lpetsc

# === Compiler and Flags ===
CFLAGS := $(CFLAGS_MODE) -I$(INC_DIR) -I$(PETSC_INC) $(CFLAGS_OMP) $(CFLAGS_MPI)
LDFLAGS := $(LDFLAGS_MODE) -L$(PETSC_LIB) $(PETSC_LIBNAME) $(LDFLAGS_OMP) $(LDFLAGS_MPI) -Wl,-rpath,$(PETSC_LIB)

# === Source and Object Files ===
SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SOURCES))

# === Build Targets ===
all: $(EXEC)

$(EXEC): $(OBJECTS) | $(BIN_DIR)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# === Directory Creation ===
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# === Convenience Targets ===
run: all
	./$(EXEC)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(EXEC)

.PHONY: all run clean
