PROJECT = bitcrasher
#CXX = clang++
CXX = g++
CXXFLAGS = -std=c++11 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
SRCS = src/libbitcrasher.cpp src/bitcrasher.cpp
RM = rm -f

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	$(RM) $(PROJECT)
