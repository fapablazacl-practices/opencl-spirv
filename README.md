
build kernel (.cl) to SPIR-V:

# Compile
clang -c -target spir -O0 -emit-llvm -o add.bc add.cl -Xclang -finclude-default-header

# Link
llvm-spirv add.bc -o add.spv
