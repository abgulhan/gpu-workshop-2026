# Build everything:   make
# Build one program:  make src/02_vector_add
# Remove binaries:    make clean
NVCC    ?= nvcc
CXX     ?= g++
# -g -lineinfo : keep source line numbers so the tools can point at your code
NVFLAGS ?= -O2 -g -lineinfo -Wno-deprecated-gpu-targets

CUDA_SRC := $(filter-out src/02_vector_add_exercise.cu src/03_bug4_inf.cu, $(wildcard src/*.cu))
CUDA_BIN := $(CUDA_SRC:.cu=)
SOL_SRC  := $(filter-out solutions/03_bug4_inf.cu, $(wildcard solutions/*.cu))
SOL_BIN  := $(SOL_SRC:.cu=)
STB      := src/stb_image.h src/stb_image_write.h

all: $(CUDA_BIN) src/03_bug4_inf src/04_blur_cpu

solutions: $(SOL_BIN) solutions/03_bug4_inf

# The debugger needs a debug build of the GPU code: -G (and no optimisation)
src/03_bug4_inf: src/03_bug4_inf.cu
	$(NVCC) -g -G -Wno-deprecated-gpu-targets -o $@ $<
solutions/03_bug4_inf: solutions/03_bug4_inf.cu
	$(NVCC) -g -G -Wno-deprecated-gpu-targets -o $@ $<

# Picture reading and writing (public-domain stb headers, fetched once if missing)
$(STB):
	curl -sSL -o $@ https://raw.githubusercontent.com/nothings/stb/master/$(notdir $@)
src/image_util.o: src/image_util.cpp src/image_util.h $(STB)
	$(CXX) -O2 -w -c -o $@ $<

src/04_blur_cpu: src/04_blur_cpu.cpp src/image_util.o
	$(CXX) -O2 -fopenmp -o $@ $< src/image_util.o

src/04_%: src/04_%.cu src/image_util.o
	$(NVCC) $(NVFLAGS) -o $@ $< src/image_util.o

%: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f $(CUDA_BIN) $(SOL_BIN) src/03_bug4_inf solutions/03_bug4_inf src/04_blur_cpu \
	      src/02_vector_add_exercise src/*.o src/out_*.png src/*.nsys-rep src/*.sqlite src/*.ncu-rep

.PHONY: all solutions clean
