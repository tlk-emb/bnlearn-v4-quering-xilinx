# bnlearn-v4-quering-xilinx

Parallel Calculation of Local Scores in Bayesian Network Structure Learning using FPGA
* n30-m9-P0 : a single-core software execution (SW)
* n30-m9-P1024 : a single-core software execution using an FPGA accelerator with parallelism 1024 (HW(P=1024))
* n30-m9-P2048 : a single-core software execution using an FPGA accelerator with parallelism 2048 (HW(P=2048))

## Usage
1. make new Application Project with template "Vector Addition"
```
File
\> New 
\> Application Project 
\> Next 
\> Select a platform from repository (xilinx_u50_gen3x16_xdma_201920_3) 
\> Application Project name (enter any names you want)
\> SW acceleration templates (Vector Addition) 
\> Finish
``` 

2. add files 
    1. replace `{Project_name}_system/{Project_name}_kernels/src/krnl_vadd.cpp` 
        -> `{Project_name}_system/{Project_name}_kernels/src/localscore_kernel.cpp`
    2. replace `{Project_name}_system/{Project_name}/src/vadd.cpp` 
    3. replace `{Project_name}_system/{Project_name}/src/vadd.h`
    4. add `{Project_name}_system/{Project_name}/src/dataset/asia30.idt` 

3. Build HW (several hours)

4. Run

## Requirement
Vitis 2020.2 + Alveo U50

# Author

* Ryota Miyagi
* Graduate School of Informatics, Kyoto University
* Email: miyagi@lab3.kuis.kyoto-u.ac.jp
