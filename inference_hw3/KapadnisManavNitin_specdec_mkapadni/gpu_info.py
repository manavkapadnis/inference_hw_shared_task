# gpu_info.py
import torch
import subprocess

def get_gpu_specs():
    """Get detailed GPU specifications"""
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        # print(f"  Max Threads per Block: {props.max_threads_per_block}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Get bandwidth using nvidia-smi
        try:
            result = subprocess.run(
                f"nvidia-smi -i {i} --query-gpu=memory.total --format=csv,nounits,noheader",
                shell=True, capture_output=True, text=True
            )
            print(f"  Details via nvidia-smi:")
            result = subprocess.run(
                f"nvidia-smi -i {i} --query-gpu=name,memory.total,compute_cap --format=csv,nounits",
                shell=True, capture_output=True, text=True
            )
            print(result.stdout)
        except:
            pass

if __name__ == "__main__":
    get_gpu_specs()
