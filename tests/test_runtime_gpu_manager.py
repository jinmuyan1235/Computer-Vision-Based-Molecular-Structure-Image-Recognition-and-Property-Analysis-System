"""Runtime GPU diagnostics should report real status without faking success."""

from src.runtime.gpu_manager import environment_status, nvidia_smi_status, tensorflow_status, torch_status
from src.runtime.inference_scheduler import InferenceScheduler


def test_runtime_status_shapes_are_stable() -> None:
    status = environment_status(run_matrix_test=False)
    assert {"python", "platform", "nvidia_smi", "torch", "tensorflow"}.issubset(status)
    assert "cuda_available" in torch_status(run_matrix_test=False)
    assert "gpu_available" in tensorflow_status(run_matrix_test=False)
    assert "gpus" in nvidia_smi_status()


def test_inference_scheduler_limits_gpu_slots() -> None:
    scheduler = InferenceScheduler(max_gpu_concurrent=1)
    with scheduler.slot_for_device("unit", "cuda:0"):
        try:
            with scheduler.slot_for_device("unit", "cuda:0"):
                raise AssertionError("second GPU slot should not be available")
        except RuntimeError as exc:
            assert "GPU 推理队列繁忙" in str(exc)
