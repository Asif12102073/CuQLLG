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
Run the demo_qllg.py in the root directory for testing and confirmation. Change the backend to CPU in demo_qllg.py if you don't have access to an nvidia GPU. This is a 10 spin Heisenberg Ring system which can also be verified using the analytical formula as the Hilbert space dimension is small enough 1024=2^10. This is the resulting Hamiltonian time evolution.  
<img width="1211" height="210" alt="image" src="https://github.com/user-attachments/assets/0b45c990-f70f-433a-ace2-9e21d789724d" />
