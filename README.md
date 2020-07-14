add llvm 10 repo: https://apt.llvm.org/

install deps:

```
sudo apt install llvm-10-dev libprotobuf-dev pybind11-dev
```

build native extension:

```
cd minjax/c
make proto
make
```

test:

```
pip install jax jaxlib pytest
pip install -e .
pytest
```

read test/test_minjax.py
