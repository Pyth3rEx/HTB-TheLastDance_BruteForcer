import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import itertools
import string
import time
import threading
import math

# Function to calculate total number of keys
def calculate_total_keys(flag_length, charset):
    return len(charset) ** flag_length

# Function to generate HTB{flag} keys
def generate_htb_keys(flag_length, charset):
    prefix = "HTB{"
    suffix = "}"
    
    for combination in itertools.product(charset, repeat=flag_length):
        flag = ''.join(combination)
        yield f"{prefix}{flag}{suffix}"

# CUDA kernel for ChaCha20 encryption with shared memory
mod = SourceModule("""
__global__ void chacha20_encrypt(char *key, char *nonce, char *message, char *output, int message_length) {
    __shared__ char shared_key[32];
    __shared__ char shared_nonce[12];

    if (threadIdx.x < 32) shared_key[threadIdx.x] = key[threadIdx.x];
    if (threadIdx.x < 12) shared_nonce[threadIdx.x] = nonce[threadIdx.x];

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < message_length) {
        output[idx] = message[idx] ^ shared_key[idx % 32] ^ shared_nonce[idx % 12];
    }
}
""")

# Global variables to track the current key and statistics
current_key = None
total_keys_tested = 0
start_time = time.time()

# Define charset and flag length
flag_length = 32
charset = string.ascii_letters + string.digits

# Calculate total keys to estimate progress
total_possible_keys = calculate_total_keys(flag_length, charset)

# Function to log the current status
def log_status():
    while True:
        if current_key is not None:
            elapsed_time = time.time() - start_time
            keys_per_second = total_keys_tested / elapsed_time
            progress = (total_keys_tested / total_possible_keys) * 100
            print(f"Currently testing key: {current_key}, Total keys tested: {total_keys_tested}, "
                  f"Keys per second: {keys_per_second:.2f}, Progress: {progress:.2f}%")
        time.sleep(5)  # Print every 5 seconds

# Start the logging thread
logging_thread = threading.Thread(target=log_status, daemon=True)
logging_thread.start()

# Main function that orchestrates the GPU-based encryption and key testing
def main():
    global current_key, total_keys_tested

    data = '''c4a66edfe80227b4fa24d431
    7aa34395a258f5893e3db1822139b8c1f04cfab9d757b9b9cca57e1df33d093f07c7f06e06bb6293676f9060a838ea138b6bc9f20b08afeb73120506e2ce7b9b9dcd9e4a421584cfaba2481132dfbdf4216e98e3facec9ba199ca3a97641e9ca9782868d0222a1d7c0d3119b867edaf2e72e2a6f7d344df39a14edc39cb6f960944ddac2aaef324827c36cba67dcb76b22119b43881a3f1262752990
    7d8273ceb459e4d4386df4e32e1aecc1aa7aaafda50cb982f6c62623cf6b29693d86b1
    457aa76ac7e2eef6cf814ae3a8d39c7
    '''

    block_size = 512
    grid_size = 32768  # Increased grid size for better GPU utilization

    encrypt_func = mod.get_function("chacha20_encrypt")

    # Create NumPy arrays for data
    key = np.frombuffer(np.random.bytes(32), dtype=np.uint8)
    nonce = np.frombuffer(np.random.bytes(12), dtype=np.uint8)
    message = np.frombuffer(data.encode(), dtype=np.uint8)

    # Allocate GPU memory
    key_gpu = cuda.mem_alloc(len(key))
    nonce_gpu = cuda.mem_alloc(len(nonce))
    message_gpu = cuda.mem_alloc(len(message))
    output_gpu = cuda.mem_alloc(len(message))

    # Copy data to GPU
    cuda.memcpy_htod(key_gpu, key)
    cuda.memcpy_htod(nonce_gpu, nonce)
    cuda.memcpy_htod(message_gpu, message)

    streams = [cuda.Stream() for _ in range(16)]  # Increased number of streams for more parallelism

    batch_size = 4096  # Increased batch size
    keys_batch = []

    for key_candidate in generate_htb_keys(flag_length, charset):
        current_key = key_candidate  # Update the global variable
        keys_batch.append(key_candidate)
        if len(keys_batch) >= batch_size:
            total_keys_tested += len(keys_batch)

            # Process the batch on the GPU
            output = np.empty_like(message)
            for stream in streams:
                encrypt_func(
                    key_gpu, nonce_gpu, message_gpu, output_gpu, np.int32(len(message)),
                    block=(block_size, 1, 1), grid=(grid_size, 1), stream=stream
                )

            for stream in streams:
                stream.synchronize()

            cuda.memcpy_dtoh(output, output_gpu)

            if output.tobytes() == data.encode():
                print(f"Match found in batch! Found key: {current_key}")
                break

            keys_batch = []

    # Free GPU memory
    output_gpu.free()
    key_gpu.free()
    nonce_gpu.free()
    message_gpu.free()

if __name__ == "__main__":
    main()
