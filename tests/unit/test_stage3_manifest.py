from mutation_forge.stage3.manifest import GATE_SPEC, load_manifest


def test_development_manifest_is_fixed_and_nonheldout() -> None:
    manifest = load_manifest("configs/manifests/stage3-development-v1.json")
    assert manifest["held_out"] is False
    assert manifest["episode_count"] == 128
    assert manifest["gate_spec"] == GATE_SPEC
