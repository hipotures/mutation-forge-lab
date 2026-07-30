from mutation_forge.stage3.config import load_stage3_config


def test_frozen_stage3_config() -> None:
    config = load_stage3_config("configs/stage3-generation.toml")
    assert config.model.name == "gpt-5.6-luna"
    assert config.model.effort == "high"
    assert config.model.slots == tuple(f"slot-{i:02d}" for i in range(8))
    assert config.model.max_repairs == 1
    assert config.experiment.episode_count == 128
    assert config.preregistration_tag == "stage3-generation-frozen-v6"
    assert config.app_server.sandbox_mode == "danger-full-access"
    assert config.app_server.approval_policy == "never"
    assert config.limits.resource_address_space_bytes == 2 * 1024 * 1024 * 1024
    assert config.limits.resource_processes == 1024
    assert config.limits.request_bytes == 65_536
    assert config.limits.response_bytes == 16_384
    assert config.limits.artifact_bytes == 32 * 1024 * 1024
