CXX=c++
CXXFLAGS=-g -Wall -Werror -std=c++14 -shared -fPIC \
	$(shell python3 -m pybind11 --includes) \
	$(shell pkg-config --cflags protobuf) \
	-I/usr/lib/llvm-10/include
LDFLAGS=$(shell pkg-config --libs protobuf) \
	$(shell llvm-config-10 --ldflags --system-libs --libs core orcjit native)

SRC=native.cc xla_data.pb.cc hlo.pb.cc
OBJ=$(SRC:.cc=.o)

jaxopt_native.so: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: proto
proto:
	protoc --cpp_out=. xla_data.proto hlo.proto

.PHONY: clean
clean:
	rm -f *.o *.pb.cc *.pb.h
