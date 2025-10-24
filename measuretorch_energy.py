import numpy as np
import torch
import argparse
from zeus.monitor import ZeusMonitor

NUM_WARMUP = 3
NUM_RUN = 100
NUM_MEASURE = 1000

def main():
    parser = argparse.ArgumentParser(description="Batch Matrix-multiplication")
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--device", type=int, default=0, 
                       help="CUDA device ID (default: 0)")
    args = parser.parse_args()
    batchsize, M, N, K = args.batchsize, args.M, args.N, args.K
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    
    # Initialize monitor with specific GPU
    monitor = ZeusMonitor(gpu_indices=[args.device], approx_instant_energy=False)
    
    # Set fixed seed for reproducibility
    np.random.seed(137)
    torch.manual_seed(137)
    torch.cuda.manual_seed(137) 

    print(f"\n{'='*60}")
    print(f"Running on: {torch.cuda.get_device_name(device)} (ID: {args.device})")
    print(f"Torch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Device capability: {torch.cuda.get_device_capability(device)}")
    print(f"Measuring the energy consumption via zeus.monitor for torch.bmm")
    print(f"Configuration: batchsize={batchsize}, M={M}, N={N}, K={K}")
    print(f"Measurements: {NUM_WARMUP} warmups, {NUM_RUN} runs × {NUM_MEASURE} operations")
    # Warmup
    print("Starting warmup...")
    for _ in range(NUM_WARMUP):
        a_torch = torch.randn(batchsize, M, K, device=device, dtype=torch.float16)
        b_torch = torch.randn(batchsize, K, N, device=device, dtype=torch.float16)
        
        monitor.begin_window("run")
        # Warmup computation
        for __ in range(NUM_MEASURE):  # Match actual measurement iteration count
            torch.bmm(a_torch, b_torch)
        energy = monitor.end_window("run")
        print(f"Energy(x{NUM_MEASURE}): {energy.gpu_energy[args.device]:.3f} J")
        del a_torch, b_torch
    print("Warmup done\n")
    
    # Main measurement runs
    results = []
    for _ in range(NUM_RUN):
        a_torch = torch.randn(batchsize, M, K, device=device, dtype=torch.float16)
        b_torch = torch.randn(batchsize, K, N, device=device, dtype=torch.float16)
        torch.bmm(a_torch, b_torch)
        torch.cuda.synchronize()
        
        # Measured computation
        monitor.begin_window("run")
        for __ in range(NUM_MEASURE):
            torch.bmm(a_torch, b_torch)
        energy = monitor.end_window("run")
        results.append(energy.gpu_energy[args.device])
        # Cleanup
        del a_torch, b_torch
        
    # Statistics
    results = np.array(results)
    mean, std = np.mean(results), np.std(results)
    print(f"Mean energy(x{NUM_MEASURE}): {mean:.3f} J (std: {std:.3f} J)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()