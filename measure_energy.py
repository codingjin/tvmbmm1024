import torch
import argparse
import os
import tvm
import numpy as np
from zeus.monitor import ZeusMonitor

NUM_WARMUP = 3
NUM_RUN = 100
NUM_MEASURE = 1000

def main():
    parser = argparse.ArgumentParser(description="Batch Matrix Multiplication Benchmark")
    parser.add_argument("--batchsize", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--M", type=int, default=4096, help="Matrix dimension M (default: 4096)")
    parser.add_argument("--N", type=int, default=4096, help="Matrix dimension N (default: 4096)")
    parser.add_argument("--K", type=int, default=4096, help="Matrix dimension K (default: 4096)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID (default: 0)")
    parser.add_argument("--funcso", type=str, default="top.so", help="funcso file to load (default: top.so)")
    args = parser.parse_args()

    np.random.seed(137)
    batchsize, M, N, K, funcso, dev_id = args.batchsize, args.M, args.N, args.K, args.funcso, args.device
    if not os.path.exists(funcso):
        print(f"Error: File '{funcso}' does not exist.")
        exit(1)
    func = tvm.runtime.load_module(funcso)
    
    device = torch.device(f"cuda:{dev_id}")
    torch.cuda.set_device(device)
    # Initialize monitor with specific GPU
    monitor = ZeusMonitor(gpu_indices=[dev_id], approx_instant_energy=False)
    
    # Show device info
    print(f"\n{'='*60}")
    print(f"Running on: {torch.cuda.get_device_name(device)} (ID: {dev_id})")
    print(f"Device capability: {torch.cuda.get_device_capability(device)}")
    print(f"Measuring the energy consumption via zeus.monitor for {funcso}")
    print(f"Configuration: batchsize={batchsize}, M={M}, N={N}, K={K}")
    print(f"Measurements: {NUM_WARMUP} warmups, {NUM_RUN} runs × {NUM_MEASURE} operations")

    # Warmup...
    print("Starting warmup...")
    for _ in range(NUM_WARMUP):
        a_nd = tvm.runtime.tensor(np.random.uniform(size=(batchsize, M, K)).astype("float16"), device=tvm.cuda(dev_id))
        b_nd = tvm.runtime.tensor(np.random.uniform(size=(batchsize, K, N)).astype("float16"), device=tvm.cuda(dev_id))
        c_nd = tvm.runtime.tensor(np.zeros((batchsize, M, N), dtype="float16"), device=tvm.cuda(dev_id))
        func(a_nd, b_nd, c_nd)
        torch.cuda.synchronize()
        
        monitor.begin_window("run")
        # Warmup computation
        for __ in range(NUM_MEASURE):  # Match actual measurement iteration count
            func(a_nd, b_nd, c_nd)
        energy = monitor.end_window("run")
        print(f"Energy(x{NUM_MEASURE}): {energy.gpu_energy[dev_id]:.3f} J")
        # Cleanup
        del a_nd, b_nd, c_nd
    print("Warmup done\n")

    # Main measurement runs
    results = []
    for _ in range(NUM_RUN):
        a_nd = tvm.runtime.tensor(np.random.uniform(size=(batchsize, M, K)).astype("float16"), device=tvm.cuda(dev_id))
        b_nd = tvm.runtime.tensor(np.random.uniform(size=(batchsize, K, N)).astype("float16"), device=tvm.cuda(dev_id))
        c_nd = tvm.runtime.tensor(np.zeros((batchsize, M, N), dtype="float16"), device=tvm.cuda(dev_id))
        func(a_nd, b_nd, c_nd)
        torch.cuda.synchronize()
        
        # Measured computation
        monitor.begin_window("run")
        for __ in range(NUM_MEASURE):
            func(a_nd, b_nd, c_nd)
        energy = monitor.end_window("run")
        results.append(energy.gpu_energy[dev_id])
        # Cleanup
        del a_nd, b_nd, c_nd
        
    # Statistics
    results = np.array(results)
    mean, std = np.mean(results), np.std(results)
    print(f"Mean energy(x{NUM_MEASURE}): {mean:.3f} J (std: {std:.3f} J)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()