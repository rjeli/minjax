CXX=c++
CXXFLAGS=-g -Wall -Werror -std=c++14 -fPIC \
	$(shell python3 -m pybind11 --includes) \
	$(shell pkg-config --cflags protobuf) \
	$(shell llvm-config-10 --cxxflags) -fexceptions
LDFLAGS=-shared \
	$(shell pkg-config --libs protobuf) \
	$(shell llvm-config-10 --ldflags --system-libs --libs core orcjit native)

SRC=main.cc xla_data.pb.cc hlo.pb.cc
OBJ=$(SRC:.cc=.o)

minjax_c.so: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: proto
proto:
	protoc --cpp_out=. xla_data.proto hlo.proto

.PHONY: clean
clean:
	rm -f *.o *.pb.cc *.pb.h
