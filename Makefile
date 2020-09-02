.SUFFIXES: .cpp

CXX = g++
CXXFLAGS = -Wall -Wextra -fopenmp -O3 -masm=intel -march=native -std=c++11 ${MKL_INC} -fno-omit-frame-pointer
LDFLAGS =  -fopenmp ${MKL_PARA}
SOURCES = utils.o dumb_kernel.o dgemm_8x8.o mkl_mult.o manual_mult.o 

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $? -o $@

main: $(SOURCES) main.o
	$(CXX) $^ ${LDFLAGS} -o $@

clean:
	$(RM) *.o *.xml


