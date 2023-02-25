```python
from numba import cuda
from matplotlib import pyplot as plt
from math import floor
import numpy as np
import time
```


```python
if cuda.is_available(): # List devices
    devices = cuda.list_devices()
    print("GPU devices:")
    for device in devices:
        print("-", device.name)
    print("Selected device:", cuda.get_current_device().name)
```

    GPU devices:
    - b'Tesla T4'
    Selected device: b'Tesla T4'
    


```python
# 1ST XP : Incrementation of an array

GRID_SIZE = 50000 # Number of blocks in the grid
BLOCKSIZE = 1024 # Number of threads for each block
```


```python
@cuda.jit # Indicate to Python that this function is a CUDA Kernel
def cudakernel0(array):
    thread_position = cuda.grid(1)
    array[thread_position] += 1
```


```python
array = np.zeros(GRID_SIZE*BLOCKSIZE, np.float32)

print("Icrementation of array by 1 | size:", GRID_SIZE*BLOCKSIZE)

# GPU
print('Kernel launch: cudakernel0[', GRID_SIZE, ', ', BLOCKSIZE,'](array)...', end=" ")
start = time.time()
cudakernel0[GRID_SIZE, BLOCKSIZE](array)
GPU_time = time.time() - start
print("Success")
      
print("Time spent for", GRID_SIZE*BLOCKSIZE, "incrementations:", GPU_time)
```

    Icrementation of array by 1 | size: 51200000
    Kernel launch: cudakernel0[ 50000 ,  1024 ](array)... 


    Success
    Time spent for 51200000 incrementation: 0.45095276832580566
    


```python
#Same xp as a function of the number of operations

operations = np.array([1 + i * 1000 for i in range(0, 50)])
size_op = operations.size
BLOCKSIZE = 1024

times = np.zeros((2, size_op))

for i in range(size_op):
    GRID_SIZE = operations[i]
    
    array = np.zeros(GRID_SIZE*BLOCKSIZE, np.float32)
    
    # GPU
    start = time.time()
    cudakernel0[GRID_SIZE, BLOCKSIZE](array)
    GPU_time = time.time() - start
    
    times[1, i] = GPU_time

plt.title("Time spent for incrementation")
plt.plot(operations * BLOCKSIZE, times[1], label = 'GPU')
plt.ylabel("Time (in s)")
plt.xlabel("Size of the array to increment")
plt.legend()
plt.show()
```


    
![png](google1.png)
    



```python
# 2ND XP : Matrix multiplication

GRID_SIZE = 50000 # Number of blocks in the grid
BLOCKSIZE = 1024 # Number of threads for each block
```


```python
@cuda.jit
def cudakernel1(matrix1, matrix2, res, blocksize):
    pos = cuda.grid(1)
    block_position = (int)(pos / blocksize)
    thread_position = pos % blocksize
    
    result = 0
        
    for i in range(blocksize):
        result += matrix1[block_position, i] * matrix2[i, thread_position]
    
    res[block_position, thread_position] = result
    
```


```python
matrix1 = np.random.rand(GRID_SIZE, BLOCKSIZE)
matrix2 = np.random.rand(BLOCKSIZE, BLOCKSIZE)

res = np.zeros((GRID_SIZE, BLOCKSIZE), np.float32)

print("Matrix multiplication, | matrix size:", matrix1.shape)

# GPU
print('Kernel launch: cudakernel1[', GRID_SIZE, ', ', BLOCKSIZE,'](array)...', end=" ")
start = time.time()
cudakernel1[GRID_SIZE, BLOCKSIZE](matrix1, matrix2, res, BLOCKSIZE)
GPU_time = time.time() - start
print("Success")
      
print("GPU time:", GPU_time)

print("Time spent to multiply matrices, 50000*1024 with 1024*1024:", GPU_time)

```

    Matrix multiplication, | matrix size: (50000, 1024)
    Kernel launch: cudakernel1[ 50000 ,  1024 ](array)... 


    Success
    GPU time: 1.4719223976135254
    Time spent to multiply a  * 1024 and a 1024*1024 matrix: 1.4719223976135254
    


```python
# Same xp as a function of the number of operations

operations = np.array([1 + i * 1000 for i in range(0, 40)])
size_op = operations.size

times = np.zeros((2, size_op))
BLOCKSIZE = 1024

for i in range(size_op):
    GRID_SIZE = operations[i]
    
    matrix1 = np.random.rand(GRID_SIZE, BLOCKSIZE)
    matrix2 = np.random.rand(BLOCKSIZE, BLOCKSIZE)

    res = np.zeros((GRID_SIZE, BLOCKSIZE), np.float32)

    # GPU
    start = time.time()
    cudakernel1[GRID_SIZE, BLOCKSIZE](matrix1, matrix2, res, BLOCKSIZE)
    GPU_time = time.time() - start
    
    times[1, i] = GPU_time

plt.title("Time spent by matrix size")
plt.plot(operations, times[1], label = 'GPU')
plt.ylabel("Time (in s)")
plt.xlabel("Size of matrices Nx1024*1024x1024")
plt.legend()
plt.show()
```


![png](google2.png)
    

