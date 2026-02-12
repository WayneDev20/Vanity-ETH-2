# Makefile for vanity-eth-address
# Works on Linux (RunPod, Ubuntu) with nvcc

NVCC = nvcc
NVCC_FLAGS = -O3 -Xptxas -v -Xcompiler -static-libgcc -Xcompiler -static-libstdc++
# Multi-arch: 52=Maxwell, 75=Turing, 86=Ampere, 89=Ada (RTX 4090)
GENCODE = -gencode arch=compute_52,code=compute_52 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_86,code=compute_86 -gencode arch=compute_89,code=compute_89
TARGET = vanity-eth-address

all: $(TARGET)

$(TARGET): src/main.cu
	$(NVCC) src/main.cu -o $(TARGET) $(NVCC_FLAGS) $(GENCODE)

clean:
	rm -f $(TARGET)

.PHONY: all clean
