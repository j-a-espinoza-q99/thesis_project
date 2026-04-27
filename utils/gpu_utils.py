"""
GPU utility functions for monitoring and optimization.
"""
import torch
import subprocess
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def get_gpu_info() -> str:
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Failed to get GPU info: {e}"


def get_optimal_batch_size(
        model: torch.nn.Module,
        sample_input_shape: Tuple[int, ...],
        start_batch_size: int = 512,
        min_batch_size: int = 16,
        memory_fraction: float = 0.85,
) -> int:
    """
    Find the largest batch size that fits in GPU memory.
    Uses binary search.
    """
    if not torch.cuda.is_available():
        return start_batch_size

    total_memory = torch.cuda.get_device_properties(0).total_memory
    target_memory = int(total_memory * memory_fraction)

    def test_batch_size(batch_size: int) -> bool:
        try:
            torch.cuda.empty_cache()
            sample_input = torch.randn(batch_size, *sample_input_shape[1:], device='cuda')
            _ = model(sample_input)
            torch.cuda.empty_cache()
            return True
        except RuntimeError:
            torch.cuda.empty_cache()
            return False

    lo, hi = min_batch_size, start_batch_size

    # Find upper bound
    while test_batch_size(hi) and hi < 4096:
        hi *= 2

    # Binary search
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if test_batch_size(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    logger.info(f"Optimal batch size: {best}")
    return best


def setup_distributed():
    """Setup distributed training if multiple GPUs are available."""
    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        logger.info(f"Distributed training: rank {local_rank}/{torch.cuda.device_count()}")
        return device, local_rank
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0


def print_gpu_summary():
    """Print a summary of GPU resources."""
    logger.info("=" * 40)
    logger.info("GPU Summary")
    logger.info("=" * 40)
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}")
        logger.info(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        logger.info(f"  Compute: {props.major}.{props.minor}")
        logger.info(f"  Multi-processor count: {props.multi_processor_count}")

    logger.info("=" * 40)