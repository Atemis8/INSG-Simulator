# === Build Configuration ===
MODE ?= opt

# === Auto-detect MPI ===
HAS_MPI := $(shell which mpicc > /dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(HAS_MPI), 1)
    CC := mpicc
    CFLAGS_MPI := # -DUSE_MPI
    # $(info MPI detected: using mpicc)
else
    CC := gcc
    CFLAGS_MPI :=
    # $(info MPI not found: using gcc)
endif

# === Auto-detect OpenMP ===
# Test if compiler supports OpenMP
HAS_OPENMP := $(shell echo 'int main(){return 0;}' | $(CC) -fopenmp -x c - -o /dev/null 2>/dev/null && echo 1 || echo 0)

ifeq ($(HAS_OPENMP), 1)
    CFLAGS_OMP := -fopenmp -DUSE_OPENMP
    LDFLAGS_OMP := -fopenmp
    # $(info OpenMP detected: enabling parallel support)
else
    CFLAGS_OMP :=
    LDFLAGS_OMP :=
    # $(info OpenMP not found: compiling without parallel support)
endif

# === Mode Configuration ===
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

# === PETSc Configuration (Auto-detect) ===
# Try to find PETSc in this order:
# 1. Environment variable PETSC_DIR
# 2. ~/petsc-install (common user install)
# 3. /usr/local/petsc (common system install)
# 4. Use pkg-config

ifeq ($(PETSC_DIR),)
    ifneq ($(wildcard $(HOME)/petsc-install/lib/petsc/conf/variables),)
        PETSC_DIR := $(HOME)/petsc-install
    else ifneq ($(wildcard /usr/local/petsc/lib/petsc/conf/variables),)
        PETSC_DIR := /usr/local/petsc
    else
        # Try pkg-config as last resort
        PETSC_DIR := $(shell pkg-config --variable=prefix PETSc 2>/dev/null)
    endif
endif

# Check if we found PETSc
ifeq ($(PETSC_DIR),)
    $(error "PETSc not found. Please set PETSC_DIR environment variable or install PETSc")
endif

PETSC_ARCH :=

# Verify PETSc installation
ifeq ($(wildcard $(PETSC_DIR)/lib/petsc/conf/variables),)
    $(error "PETSc installation incomplete at $(PETSC_DIR). Missing conf/variables file")
endif

# Get PETSc variables (includes, libraries, etc.)
include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

# === Compiler and Flags ===
CFLAGS := $(CFLAGS_MODE) $(CFLAGS_MPI) $(CFLAGS_OMP) -I$(INC_DIR) $(PETSC_CC_INCLUDES)
LDFLAGS := $(LDFLAGS_MODE) $(LDFLAGS_OMP) $(PETSC_LIB)

# === Source and Object Files ===
SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SOURCES))

# === Build Targets ===
all: $(EXEC)

.DEFAULT_GOAL := all

$(EXEC): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# === Directory Creation ===
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# === Convenience Targets ===
run: all
	./$(EXEC)

clean::
	rm -rf $(OBJ_DIR) $(EXEC)

info:
	@echo "=== Build Configuration ==="
	@echo "Mode: $(MODE)"
	@echo "MPI: $(HAS_MPI) (Compiler: $(CC))"
	@echo "OpenMP: $(HAS_OPENMP)"
	@echo "PETSc Directory: $(PETSC_DIR)"
	@echo "PETSc Arch: $(PETSC_ARCH)"
	@echo ""
	@echo "=== Compiler Flags ==="
	@echo "CFLAGS: $(CFLAGS)"
	@echo ""
	@echo "=== Linker Flags ==="
	@echo "LDFLAGS: $(LDFLAGS)"

.PHONY: all run clean info

# TO RUN MPI OFFLINE 
# sudo ifconfig bridge0 up
# sudo ifconfig bridge0 inet 10.0.0.1/24 add
# ifconfig bridge0 => inet 10.0.0.1 netmask 0xffffff00
#