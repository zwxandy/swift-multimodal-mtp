from swift.callbacks import activation_cpu_offload


def test_activation_cpu_offload_imports_without_fsdp2():
    assert isinstance(activation_cpu_offload.FSDP_TYPES, tuple)
    assert activation_cpu_offload.FSDP in activation_cpu_offload.FSDP_TYPES
