import tvm
import argparse
from tvm import meta_schedule as ms
from tvm import te
from tvm.meta_schedule.runner.config import EvaluatorConfig
from tvm.script import tir as T
from typing import Tuple
from tvm.meta_schedule.testing import te_workload
from tvm.te import create_prim_func
from pathlib import Path
import subprocess

#target = tvm.target.Target(f"cuda -max_threads_per_block 1024 -max_shared_memory_per_block 49152") # 3090
#target = tvm.target.Target({"kind": "cuda", "arch": "sm_86", "max_threads_per_block": 1024, "max_shared_memory_per_block": 49152}) # 3090
#target = tvm.target.Target({"kind": "cuda", "arch": "sm_70", "max_threads_per_block": 1024, "max_shared_memory_per_block": 49152}) # V100
#target = tvm.target.Target({"kind": "cuda", "arch": "sm_80", "max_threads_per_block": 1024, "max_shared_memory_per_block": 49152}) # A100

FILE_RUNSECS = "run_secs"
topso = "./top.so"
lastso = "./last.so"
FILE_TURNINGRECORDS = "turningrecords"

def batch_matmul_mkkn(  # pylint: disable=invalid-name,missing-docstring
    B: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str = "float16",
    out_dtype: str = "float16",
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    x = te.placeholder((B, M, K), name="X", dtype=in_dtype)
    y = te.placeholder((B, K, N), name="Y", dtype=in_dtype)
    k = te.reduce_axis((0, K), name="k")
    z = te.compute(  # pylint: disable=invalid-name
        (B, M, N),
        lambda b, i, j: te.sum(
            x[b][i][k].astype(out_dtype) * y[b][k][j].astype(out_dtype),
            axis=[k],
        ),
        name="Z",
    )
    return (x, y, z)

def get_gpu_name():
    """
    Uses nvidia-smi to get the GPU name.
    Returns a short name: "3090", "4090", "V100", or "A100"
    """
    try:
        # Run nvidia-smi and get the GPU name
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        full_name = result.stdout.strip()
        # Map full GPU names to our short names
        if "3090" in full_name:
            return "3090"
        elif "4090" in full_name:
            return "4090"
        elif "V100" in full_name:
            return "V100"
        elif "A100" in full_name:
            return "A100"
        else:
            raise ValueError(f"Unrecognized GPU: {full_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to detect GPU: {e}")


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

    args = parser.parse_args()
    batchsize, M, N, K, GPU = args.batchsize, args.M, args.N, args.K, get_gpu_name()
    print(f"Batch Matmul, GPU model: {GPU}")
    print(f"Batch size: {batchsize}, M: {M}, N: {N}, K: {K}")

    if GPU == "3090":
        target = tvm.target.Target({
            "kind": "cuda",
            "arch": "sm_86",
            "max_threads_per_block": 1024,
            "max_shared_memory_per_block": 49152
        })
    elif GPU == "4090":
        target = tvm.target.Target({
            "kind": "cuda",
            "arch": "sm_89",
            "max_threads_per_block": 1024,
            "max_shared_memory_per_block": 49152
        })
    elif GPU == "V100":
        target = tvm.target.Target({
            "kind": "cuda",
            "arch": "sm_70",
            "max_threads_per_block": 1024,
            "max_shared_memory_per_block": 49152
        })
    elif GPU == "A100":
        target = tvm.target.Target({
            "kind": "cuda",
            "arch": "sm_80",
            "max_threads_per_block": 1024,
            "max_shared_memory_per_block": 49152
        })
    else:
        raise ValueError(f"Unknown GPU: {GPU}")
        exit(1)
    print(f"target: {target}")

    bmm = create_prim_func(batch_matmul_mkkn(batchsize, M, K, N, in_dtype="float16", out_dtype="float16"))
    print(bmm)

    database = ms.tune_tir(
        mod=bmm,
        target=target,
        max_trials_global=1000,
        num_trials_per_iter=64,
        work_dir="./",
        runner=ms.runner.LocalRunner(
            evaluator_config=EvaluatorConfig(
                number=10,
                enable_cpu_cache_flush=False,
            )
        )
    )
    
    tune_record_list = database.get_all_tuning_records()
    totalnum = len(tune_record_list)
    workload = tune_record_list[0].workload
    mod = workload.mod
    print(f"The total number of tuning records is {totalnum}")

    sortedrecords = database.get_top_k(workload, totalnum)
    Path(FILE_TURNINGRECORDS).write_text("")
    with open(FILE_TURNINGRECORDS, "a") as f:
        for r in tune_record_list:
            f.write(f"{r.run_secs[0].value}\n")

    Path(FILE_RUNSECS).write_text("")
    with open(FILE_RUNSECS, "a") as f:
        # top so
        db = ms.database.MemoryDatabase()
        db.commit_workload(sortedrecords[0].workload.mod)
        db.commit_tuning_record(sortedrecords[0])
        sch = ms.tir_integration.compile_tir(db, mod, target)
        print("Topsch mod")
        print(sch.mod)
        with tvm.transform.PassContext(config={"tir.disable_assert": True}):
            lib = tvm.tir.build(sch.mod, target)
        lib.export_library(topso)
        f.write(f"{sortedrecords[0].run_secs[0].value}\n")

        # last so
        db = ms.database.MemoryDatabase()
        db.commit_workload(sortedrecords[-1].workload.mod)
        db.commit_tuning_record(sortedrecords[-1])
        sch = ms.tir_integration.compile_tir(db, mod, target)
        print("Lastsch mod")
        print(sch.mod)
        with tvm.transform.PassContext(config={"tir.disable_assert": True}):
            lib = tvm.tir.build(sch.mod, target)
        lib.export_library(lastso)
        f.write(f"{sortedrecords[-1].run_secs[0].value}\n")

if __name__ == "__main__":
    main()


