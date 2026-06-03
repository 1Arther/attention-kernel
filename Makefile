NVCC ?= nvcc
ARCH ?= sm_86
TARGET = attention_bench

SRC = main.cu attention_kernel.cu

all:
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o