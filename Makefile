PROJECT = bitcrasher
#CXX = clang++
CXX = g++
CXXFLAGS = -std=c++11 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
SRCS = src/ADMM.cpp src/order_basis.cpp src/DCT_function.cpp src/cholesky.cpp src/im2col.cpp src/main.cpp

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
