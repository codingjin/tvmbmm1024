import numpy as np
import torch
import argparse
import pynvml

NUM_WARMUP = 3
NUM_RUN = 100
NUM_MEASURE = 1000

def main():
    parser = argparse.ArgumentParser(description="Batch Matrix-multiplication")
    parser.add_argument(
        "--batchsize", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--M", type=int, default=4096,
        help="Matrix dimension M (default: 4096)"
    )
    parser.add_argument(
        "--N", type=int, default=4096,
        help="Matrix dimension N (default: 4096)"
    )
    parser.add_argument(
        "--K", type=int, default=4096,
        help="Matrix dimension K (default: 4096)"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="CUDA device ID (default: 0)"
    )
    args = parser.parse_args()
    batchsize, M, N, K, device_id = args.batchsize, args.M, args.N, args.K, args.device

    # Set random seeds
    np.random.seed(137)
    torch.manual_seed(137)
    torch.cuda.manual_seed(137)

    # Set the specific CUDA device
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Initialize NVML and get handle for the specified device only
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    print(f"\n{'='*60}")
    print(f"Running on: {torch.cuda.get_device_name(device)} (ID: {args.device})")
    print(f"Torch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Device capability: {torch.cuda.get_device_capability(device)}")
    print(f"Measuring the energy consumption via pynvml for torch.bmm")
    print(f"Configuration: batchsize={batchsize}, M={M}, N={N}, K={K}")
    print(f"Measurements: {NUM_WARMUP} warmups, {NUM_RUN} runs × {NUM_MEASURE} operations")

    # warmup
    print(f"Starting warmup...")
    for i in range(NUM_WARMUP):
        a_torch = torch.randn(batchsize, M, K, device=device, dtype=torch.float16)
        b_torch = torch.randn(batchsize, K, N, device=device, dtype=torch.float16)
        torch.bmm(a_torch, b_torch)
        torch.cuda.synchronize()

        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        
        for _ in range(NUM_MEASURE):
            torch.bmm(a_torch, b_torch)
        
        torch.cuda.synchronize()
        consumed_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - start_energy
        total_energy = consumed_energy * 0.001  # Convert mJ to J
        print(f"Energy(x{NUM_MEASURE}): {total_energy:.3f} J")
        del a_torch, b_torch
    print("Warmup done\n")
    
    # run
    results = []
    for _ in range(NUM_RUN):
        a_torch = torch.randn(batchsize, M, K, device=device, dtype=torch.float16)
        b_torch = torch.randn(batchsize, K, N, device=device, dtype=torch.float16)
        torch.bmm(a_torch, b_torch)
        torch.cuda.synchronize()
        
        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

        for __ in range(NUM_MEASURE):
            torch.bmm(a_torch, b_torch)
        
        torch.cuda.synchronize()
        consumed_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - start_energy
        total_energy = consumed_energy * 0.001  # Convert mJ to J
        results.append(total_energy)
        del a_torch, b_torch
    
    pynvml.nvmlShutdown()
    # Statistics
    results = np.array(results)
    mean = np.mean(results)
    std = np.std(results)
    
    print(f"\n{'='*60}")
    print(f"Configuration: batchsize={batchsize}, M={M}, N={N}, K={K}")
    print(f"Measurements: {NUM_RUN} runs × {NUM_MEASURE} operations")
    print(f"Mean energy(x{NUM_MEASURE}): {mean:.3f} J (std: {std:.3f} J)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()