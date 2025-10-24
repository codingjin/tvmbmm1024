import torch
import argparse
import os
import tvm
import numpy as np

NUM_WARMUP = 10
NUM_RUN = 1000

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
    torch.manual_seed(137)
    torch.cuda.manual_seed_all(137)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batchsize, M, N, K, funcso, dev_id = args.batchsize, args.M, args.N, args.K, args.funcso, args.device
    if not os.path.exists(funcso):
        print(f"Error: File '{funcso}' does not exist.")
        exit(1)
    func = tvm.runtime.load_module(funcso)
    
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    a_np = np.random.uniform(size=(batchsize, M, K)).astype("float16")
    b_np = np.random.uniform(size=(batchsize, K, N)).astype("float16")
    a_nd = tvm.runtime.tensor(a_np, device=tvm.cuda(dev_id))
    b_nd = tvm.runtime.tensor(b_np, device=tvm.cuda(dev_id))
    c_nd = tvm.runtime.tensor(np.zeros((batchsize, M, N), dtype="float16"), device=tvm.cuda(dev_id))

    # Show device info
    print("----------------------------------------------")
    print(f"Running on: {torch.cuda.get_device_name(device)} (ID: {args.device})")
    print(f"Device capability: {torch.cuda.get_device_capability(device)}")
    print(f"Benchmarking {funcso}")
    print(f"Configuration: batchsize: {batchsize}, M={M}, N={N}, K={K}")
    print(f"{NUM_WARMUP} warmups, {NUM_RUN} runs")
    print("----------------------------------------------")

    # Warmup...
    print("Starting warmup...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(NUM_WARMUP):
        func(a_nd, b_nd, c_nd)
    end_event.record()
    torch.cuda.synchronize(device)
    print("Warmup complete\n")
    
    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(NUM_RUN):
        func(a_nd, b_nd, c_nd)
    end_event.record()
    torch.cuda.synchronize(device)
    timems = start_event.elapsed_time(end_event)  # ms

    flops = 2 * batchsize * M * N * K
    gflops = flops * NUM_RUN * 1e-6 / timems
    """
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_s = elapsed_ms / 1e3               # convert ms → s
    total_flops = 2 * batchsize * M * N * K * NUM_RUN
    gflops = total_flops / elapsed_s / 1e9     # convert FLOPs → GFLOPs
    """
    print(f"Performance: {int(round(gflops))} GFLOPs")
    print("----------------------------------------------")

if __name__ == "__main__":
    main()