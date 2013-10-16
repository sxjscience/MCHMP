CC=gcc
CFLAGS=-O3 -c -Wall -fopenmp -I/usr/local/include/eigen3 -I/usr/local/include 
LDFLAGS=-lstdc++ -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lgomp -lgsl
SOURCES=main.cpp ksvd.cpp ml_common.cpp ml_graphic_common.cpp ml_graphic.cpp ml_hmp.cpp ml_io.cpp ml_openmp_common.cpp ml_random.cpp ml_omp.cpp ml_helper.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=hmp

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
