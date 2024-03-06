# C++ fordító beállítások
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# Célfájl és függőségek
all: tsp

# Cél: tsp futtatható fájl létrehozása
tsp: tsp.o kernel_loader.o
	$(CXX) $(CXXFLAGS) tsp.o kernel_loader.o -o tsp -lOpenCL

# Cél: tsp.o objektumfájl létrehozása
tsp.o: tsp.cpp
	$(CXX) $(CXXFLAGS) $(OPENCL_INCLUDE) -c tsp.cpp -o tsp.o 

# Cél: kernel_loader.o objektumfájl létrehozása
kernel_loader.o: src/kernel_loader/kernel_loader.cpp src/kernel_loader/kernel_loader.hpp
	$(CXX) $(CXXFLAGS) $(OPENCL_INCLUDE) -c src/kernel_loader/kernel_loader.cpp -o kernel_loader.o

# Tisztítás célja: törölje az objektumfájlokat és a futtatható fájlt
clean:
	rm -f *.o tsp