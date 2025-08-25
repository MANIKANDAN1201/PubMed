from typing import Tuple
from benchmark_retrieval import evaluate


def run_bioasq_benchmark(bioasq_path: str, model_name: str, top_k: int = 5) -> Tuple[float, float]:
    return evaluate(bioasq_path, model_name, top_k=top_k)


