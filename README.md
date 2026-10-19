# CuQLLG
CuPy based Quantum-LLG solver library. This is a CuPy based numerical Quantum-LLG equation solver originally developed by Vahid Azimi et al.  in MATLAB. This is a reimplentation of that solver but using CuBLAS , CuSolver and Numba JIT compiler ( if the backend is CPU) .  This is the LREI algorithm implementation( version 0.10 comes only with RK4 ) 


## Installation 

Install GitCLI, miniconda/ conda on your system. create a new environment : 

```bash
conda create -n CuQLLG python=3.12 
```
Install optional Cupy for GPU accelrated backend if you have an Nvidia GPU (Driver 580+ and CUDA Toolkit installed on your system)

```bash
conda activate CuQLLG
#CUDA version auto handled by conda
conda install -c conda-forge cupy
```

Git clone the repo into your desired directory
```bash
git clone https://github.com/Asif12102073/CuQLLG/tree/main CuQLLG 
```

Navigate to the root directory of the cloned repo. 
```bash
cd CuQLLG
```

Finally pip install the dependencies
```bash
pip install . 
```

## Testing
### 10-spin Heisenberg ring test

Run the demo_qllg.py in the root directory for testing and confirmation. Change the backend to CPU in demo_qllg.py if you don't have access to an nvidia GPU. This is a 10 spin Heisenberg Ring system which can also be verified using the analytical formula as the Hilbert space dimension is small enough 1024=2^10. This is the resulting Hamiltonian time evolution.  
<img width="762" height="186" alt="image" src="https://github.com/user-attachments/assets/66ec0c60-f0b8-4231-86c4-f559a840f39e" />

### Cross-validation against classical and dense-matrix methods. 

Run the demo2.py for a comparison test against classical and quantum llg dynamics. The classical llg follows the same trend but underpredicts against the quantum version which includes the micromotion of atomistic spin effects. It also produces plots on how stiff systems can potentially increase errors in qllg solvers. So, time step may need to be reduced to avoid that .

<img width="1950" height="1350" alt="qllg_comparison3" src="https://github.com/user-attachments/assets/53f4f99c-1def-4a2c-99d4-2aa6bc8a01c1" />


## Future Directions 

The library will be updated with the AB method alongside the RK4 solver. More efficient optimization will be added if possible.
