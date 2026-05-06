# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryProfile:
    components_gb: dict[str, dict[int, float]]

    def total_gb(self, parallelism: int) -> float:
        missing = [name for name, values in self.components_gb.items() if parallelism not in values]
        if missing:
            raise KeyError(f"Missing memory profile for parallelism {parallelism}: {', '.join(missing)}")
        return sum(values[parallelism] for values in self.components_gb.values())


@dataclass(frozen=True)
class PEHardwareProfile:
    ttft_ms: dict[int, dict[int, dict[int, float]]]
    tpot_ms: dict[int, dict[int, dict[int, float]]]
    memory: MemoryProfile
    source_note: str
    sim_ttft_ms: dict[int, dict[int, dict[int, float]]] | None = None
    sim_tpot_ms: dict[int, dict[int, dict[int, float]]] | None = None
    init_time_s: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PEModelProfile:
    name: str
    kv_cache_per_token_gb: dict[int, float]
    hardware: dict[str, PEHardwareProfile]


@dataclass(frozen=True)
class StageHardwareProfile:
    latency_s: dict[int, float]
    memory: MemoryProfile
    source_note: str


@dataclass(frozen=True)
class GeneratorModelProfile:
    name: str
    encoder_cpu_latency_s: float
    encoder_cpu_memory: MemoryProfile
    main_dit_bundle_size: int
    encoder: dict[str, StageHardwareProfile]
    dit: dict[str, StageHardwareProfile]
    vae: dict[str, StageHardwareProfile]
    init_time_s: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TDProfileData:
    pe_models: dict[str, PEModelProfile]
    generator_models: dict[str, GeneratorModelProfile]


def _ms_to_s(values: dict[int, float]) -> dict[int, float]:
    return {parallelism: latency_ms / 1000.0 for parallelism, latency_ms in values.items()}


def _generator_hardware_profiles(
    a800_latency_s: dict[int, float],
    h200_latency_ms: dict[int, float],
    memory_components_gb: dict[str, dict[int, float]],
    stage_name: str,
) -> dict[str, StageHardwareProfile]:
    memory = MemoryProfile(memory_components_gb)
    return {
        "A800": StageHardwareProfile(
            latency_s=a800_latency_s,
            memory=memory,
            source_note=f"{stage_name} A800 NVLink latency and memory",
        ),
        "A100": StageHardwareProfile(
            latency_s=a800_latency_s,
            memory=memory,
            source_note=f"{stage_name} A800 NVLink latency and memory reused for A100",
        ),
        "H100": StageHardwareProfile(
            latency_s=_ms_to_s(h200_latency_ms),
            memory=memory,
            source_note=f"{stage_name} H200 NVLink latency reused for H100; A800 memory",
        ),
    }


def build_default_profile_data() -> TDProfileData:
    pe7b_init_time_s = {8: 20.982828, 4: 19.974589, 2: 20.963828, 1: 19.95004}
    pe7b_a800_memory = MemoryProfile(
        {
            "weights": {1: 13.99, 2: 7.01, 4: 3.51, 8: 1.76},
            "cudagraph": {1: 0.84, 2: 0.96, 4: 1.49, 8: 1.47},
            "others": {1: 0.99, 2: 1.04, 4: 2.56, 8: 2.59},
        }
    )
    pe7b_a800_simu_ttft = {
        128: {
            512: {1: 25.999, 2: 15.743, 4: 10.718, 8: 8.348},
            2048: {1: 25.999, 2: 15.743, 4: 10.718, 8: 8.348},
        },
        256: {
            256: {1: 38.174, 2: 22.974, 4: 18.546, 8: 11.868},
            1920: {1: 38.174, 2: 22.974, 4: 18.546, 8: 11.868},
        },
        384: {
            128: {1: 52.593, 2: 30.118, 4: 22.154, 8: 19.118},
            1792: {1: 52.593, 2: 30.118, 4: 22.154, 8: 19.118},
        },
        512: {1664: {1: 67.7, 2: 38.885, 4: 28.39, 8: 18.214}},
        640: {1536: {1: 81.153, 2: 46.833, 4: 31.232, 8: 20.766}},
        768: {1408: {1: 94.462, 2: 54.669, 4: 34.093, 8: 23.306}},
        896: {1280: {1: 110.681, 2: 63.625, 4: 41.49, 8: 26.915}},
        1024: {1152: {1: 126.853, 2: 72.543, 4: 48.894, 8: 30.521}},
        1152: {1024: {1: 142.843, 2: 81.277, 4: 53.449, 8: 33.888}},
        1280: {896: {1: 158.662, 2: 89.91, 4: 57.931, 8: 37.21}},
        1408: {768: {1: 174.365, 2: 98.475, 4: 62.364, 8: 40.502}},
        1536: {640: {1: 190.011, 2: 107.006, 4: 66.773, 8: 43.78}},
        1664: {512: {1: 206.242, 2: 116.236, 4: 71.664, 8: 47.127}},
        1792: {384: {1: 222.232, 2: 125.338, 4: 76.455, 8: 50.453}},
        1920: {256: {1: 238.062, 2: 134.353, 4: 81.178, 8: 53.765}},
        2048: {128: {1: 253.812, 2: 143.326, 4: 85.868, 8: 57.069}},
    }
    pe7b_a800_simu_tpot = {
        128: {
            512: {1: 11.334, 2: 6.58, 4: 4.09, 8: 2.823},
            2048: {1: 11.501, 2: 6.731, 4: 4.243, 8: 2.984},
        },
        256: {
            256: {1: 11.334, 2: 6.58, 4: 4.09, 8: 2.823},
            1920: {1: 11.503, 2: 6.733, 4: 4.243, 8: 2.985},
        },
        384: {
            128: {1: 11.343, 2: 6.585, 4: 4.095, 8: 2.843},
            1792: {1: 11.505, 2: 6.736, 4: 4.244, 8: 2.986},
        },
        512: {1664: {1: 11.506, 2: 6.738, 4: 4.245, 8: 2.987}},
        640: {1536: {1: 11.508, 2: 6.74, 4: 4.245, 8: 2.987}},
        768: {1408: {1: 11.51, 2: 6.742, 4: 4.246, 8: 2.988}},
        896: {1280: {1: 11.512, 2: 6.744, 4: 4.246, 8: 2.989}},
        1024: {1152: {1: 11.513, 2: 6.746, 4: 4.247, 8: 2.99}},
        1152: {1024: {1: 11.515, 2: 6.749, 4: 4.248, 8: 2.99}},
        1280: {896: {1: 11.517, 2: 6.751, 4: 4.248, 8: 2.991}},
        1408: {768: {1: 11.519, 2: 6.753, 4: 4.249, 8: 2.992}},
        1536: {640: {1: 11.52, 2: 6.755, 4: 4.25, 8: 2.993}},
        1664: {512: {1: 11.522, 2: 6.757, 4: 4.25, 8: 2.994}},
        1792: {384: {1: 11.524, 2: 6.759, 4: 4.251, 8: 2.994}},
        1920: {256: {1: 11.526, 2: 6.762, 4: 4.252, 8: 2.995}},
        2048: {128: {1: 11.534, 2: 6.762, 4: 4.253, 8: 2.995}},
    }
    pe7b_h100_latency_ttft = {
        128: {
            512: {1: 12.965, 2: 8.275, 4: 6.561, 8: 6.087},
            2048: {1: 12.965, 2: 8.275, 4: 6.561, 8: 6.087},
        },
        256: {
            256: {1: 15.141, 2: 10.262, 4: 8.591, 8: 7.91},
            1920: {1: 15.141, 2: 10.262, 4: 8.591, 8: 7.91},
        },
        384: {
            128: {1: 19.053, 2: 12.66, 4: 10.36, 8: 9.622},
            1792: {1: 19.053, 2: 12.66, 4: 10.36, 8: 9.622},
        },
        512: {1664: {1: 22.923, 2: 15.084, 4: 12.624, 8: 11.458}},
        640: {1536: {1: 28.533, 2: 18.275, 4: 14.898, 8: 13.245}},
        768: {1408: {1: 34.063, 2: 21.391, 4: 17.143, 8: 15.017}},
        896: {1280: {1: 38.963, 2: 24.68, 4: 19.469, 8: 16.885}},
        1024: {1152: {1: 43.837, 2: 27.945, 4: 21.786, 8: 18.748}},
        1152: {1024: {1: 49.485, 2: 31.254, 4: 23.818, 8: 20.299}},
        1280: {896: {1: 55.067, 2: 34.561, 4: 25.889, 8: 21.825}},
        1408: {768: {1: 60.605, 2: 37.867, 4: 27.985, 8: 23.334}},
        1536: {640: {1: 66.121, 2: 41.173, 4: 30.093, 8: 24.834}},
        1664: {512: {1: 71.878, 2: 44.614, 4: 32.012, 8: 27.306}},
        1792: {384: {1: 77.554, 2: 48.001, 4: 33.905, 8: 30.072}},
        1920: {256: {1: 83.176, 2: 51.351, 4: 35.78, 8: 33.033}},
        2048: {128: {1: 88.771, 2: 54.682, 4: 37.645, 8: 36.092}},
    }
    pe7b_h100_latency_tpot = {
        128: {
            512: {1: 6.221, 2: 3.989, 4: 2.701, 8: 2.065},
            2048: {1: 6.251, 2: 4.02, 4: 2.719, 8: 2.131},
        },
        256: {
            256: {1: 6.221, 2: 3.989, 4: 2.701, 8: 2.065},
            1920: {1: 6.253, 2: 4.02, 4: 2.72, 8: 2.132},
        },
        384: {
            128: {1: 6.243, 2: 4.017, 4: 2.725, 8: 2.088},
            1792: {1: 6.255, 2: 4.021, 4: 2.722, 8: 2.132},
        },
        512: {1664: {1: 6.256, 2: 4.022, 4: 2.723, 8: 2.132}},
        640: {1536: {1: 6.258, 2: 4.022, 4: 2.724, 8: 2.132}},
        768: {1408: {1: 6.26, 2: 4.023, 4: 2.725, 8: 2.132}},
        896: {1280: {1: 6.262, 2: 4.024, 4: 2.726, 8: 2.133}},
        1024: {1152: {1: 6.263, 2: 4.024, 4: 2.727, 8: 2.133}},
        1152: {1024: {1: 6.265, 2: 4.025, 4: 2.729, 8: 2.133}},
        1280: {896: {1: 6.267, 2: 4.025, 4: 2.73, 8: 2.133}},
        1408: {768: {1: 6.269, 2: 4.026, 4: 2.731, 8: 2.133}},
        1536: {640: {1: 6.27, 2: 4.027, 4: 2.732, 8: 2.133}},
        1664: {512: {1: 6.272, 2: 4.027, 4: 2.733, 8: 2.134}},
        1792: {384: {1: 6.274, 2: 4.028, 4: 2.734, 8: 2.134}},
        1920: {256: {1: 6.275, 2: 4.029, 4: 2.736, 8: 2.134}},
        2048: {128: {1: 6.277, 2: 4.029, 4: 2.737, 8: 2.134}},
    }
    pe7b = PEModelProfile(
        name="pe7b",
        kv_cache_per_token_gb={1: 0.0001220916282, 2: 0.00006103920126, 4: 0.00003051310889, 8: 0.00001525755294},
        hardware={
            "A800": PEHardwareProfile(
                ttft_ms={
                    128: {
                        512: {1: 25.95552, 2: 30.38936, 4: 27.57219, 8: 35.69319},
                        2048: {1: 27.43913, 2: 29.94049, 4: 28.43359, 8: 30.51378},
                    }
                },
                tpot_ms={
                    128: {
                        512: {1: 10.89454, 2: 6.867308, 4: 4.491277, 8: 3.226533},
                        2048: {1: 10.94458, 2: 6.880903, 4: 4.501845, 8: 3.240627},
                    }
                },
                memory=pe7b_a800_memory,
                source_note="PE7B A800 NVLink latency and memory",
                sim_ttft_ms=pe7b_a800_simu_ttft,
                sim_tpot_ms=pe7b_a800_simu_tpot,
                init_time_s=pe7b_init_time_s,
            ),
            "A100": PEHardwareProfile(
                ttft_ms={
                    128: {
                        512: {1: 30.08084, 2: 32.51202, 4: 33.03063, 8: 33.737994},
                        2048: {1: 28.51644, 2: 33.00214, 4: 34.86355, 8: 36.039466},
                    }
                },
                tpot_ms={
                    128: {
                        512: {1: 12.89123, 2: 7.959073, 4: 5.031289, 8: 3.864208},
                        2048: {1: 12.95548, 2: 7.984124, 4: 5.04194, 8: 3.864316},
                    }
                },
                memory=MemoryProfile(
                    {
                        "weights": {1: 13.99, 2: 7.01, 4: 3.51, 8: 1.76},
                        "cudagraph": {1: 0.39, 2: 0.46, 4: 1.36, 8: 1.47},
                        "others": {1: 0.8, 2: 0.98, 4: 1.77, 8: 2.59},
                    }
                ),
                source_note="PE7B A100 NVLink latency; A800 memory reused for TP8",
                init_time_s=pe7b_init_time_s,
            ),
            "H100": PEHardwareProfile(
                ttft_ms=pe7b_h100_latency_ttft,
                tpot_ms=pe7b_h100_latency_tpot,
                memory=pe7b_a800_memory,
                source_note="PE7B H100 NVLink simulated latency; A800 memory",
                sim_ttft_ms=pe7b_h100_latency_ttft,
                sim_tpot_ms=pe7b_h100_latency_tpot,
                init_time_s=pe7b_init_time_s,
            ),
        },
    )

    wan22_encoder_memory = MemoryProfile(
        {
            "weights": {1: 21.16, 2: 10.58, 4: 5.29, 8: 2.65},
            "runtime": {1: 0.7926, 2: 3.3882, 4: 2.0102, 8: 1.2284},
        }
    )
    wan22 = GeneratorModelProfile(
        name="wan2.2-ti2v-5b",
        encoder_cpu_latency_s=6.769752,
        encoder_cpu_memory=wan22_encoder_memory,
        main_dit_bundle_size=8,
        encoder=_generator_hardware_profiles(
            {1: 0.1641, 2: 0.1487, 4: 0.1701, 8: 0.183},
            {1: 48, 2: 62, 4: 244.882, 8: 352.391},
            wan22_encoder_memory.components_gb,
            "WAN2.2 Encoder",
        ),
        dit=_generator_hardware_profiles(
            {1: 253.5454, 2: 141.4324, 4: 73.7484, 8: 43.1319},
            {1: 97350, 2: 59990, 4: 27346.614, 8: 14536.431},
            {
                "weights": {1: 9.31, 2: 9.31, 4: 9.31, 8: 9.31},
                "runtime": {1: 7.4605, 2: 6.1499, 4: 3.854, 8: 3.1229},
            },
            "WAN2.2 DiT",
        ),
        vae=_generator_hardware_profiles(
            {1: 25.4299, 2: 12.8685, 4: 6.6103, 8: 3.5325},
            {1: 9410, 2: 5240, 4: 2865.301, 8: 1537.965},
            {
                "weights": {1: 2.63, 2: 2.63, 4: 2.63, 8: 2.63},
                "runtime": {1: 26.9, 2: 17.9488, 4: 21.8891, 8: 19.5799},
            },
            "WAN2.2 VAE",
        ),
        init_time_s={1: 24.447095, 2: 28.080947, 4: 40.251256, 8: 61.020383},
    )

    wan21_encoder_memory = MemoryProfile(
        {
            "weights": {1: 21.16, 2: 10.58, 4: 5.29, 8: 2.65},
            "runtime": {1: 0.5960, 2: 2.1330, 4: 1.4719, 8: 1.2154},
        }
    )
    wan21 = GeneratorModelProfile(
        name="wan2.1-t2v-1.3b",
        encoder_cpu_latency_s=6.769752,
        encoder_cpu_memory=wan22_encoder_memory,
        main_dit_bundle_size=4,
        encoder=_generator_hardware_profiles(
            {1: 0.1616, 2: 0.1494, 4: 0.1464, 8: 0.1507},
            {1: 239.965, 2: 160.882, 4: 130.643, 8: 109.543},
            wan21_encoder_memory.components_gb,
            "WAN2.1 Encoder",
        ),
        dit=_generator_hardware_profiles(
            {1: 133.6284, 2: 76.0782, 4: 40.6529, 8: 24.2475},
            {1: 44886.895, 2: 25820.131, 4: 13395.720, 8: 9042.033},
            {
                "weights": {1: 2.64, 2: 2.64, 4: 2.64, 8: 2.64},
                "runtime": {1: 2.4398, 2: 2.4065, 4: 1.4719, 8: 1.2154},
            },
            "WAN2.1 DiT",
        ),
        vae=_generator_hardware_profiles(
            {1: 6.0267, 2: 3.4496, 4: 1.8910, 8: 1.0937},
            {1: 2920.790, 2: 1669.970, 4: 917.065, 8: 622.574},
            {
                "weights": {1: 0.27, 2: 0.27, 4: 0.27, 8: 0.27},
                "runtime": {1: 10.7796, 2: 7.8264, 4: 5.9074, 8: 6.0826},
            },
            "WAN2.1 VAE",
        ),
    )

    z_image_encoder_memory = MemoryProfile(
        {
            "weights": {1: 15.0, 2: 15.0, 4: 15.0, 8: 15.0},
            "runtime": {1: 0.6597, 2: 0.6597, 4: 0.6597, 8: 0.6597},
        }
    )
    z_image = GeneratorModelProfile(
        name="z-image",
        encoder_cpu_latency_s=6.769752,
        encoder_cpu_memory=wan22_encoder_memory,
        main_dit_bundle_size=2,
        encoder=_generator_hardware_profiles(
            {1: 0.2524, 2: 0.2525, 4: 0.2481, 8: 0.2540},
            {1: 214.139, 2: 218.791, 4: 215.696, 8: 216.209},
            z_image_encoder_memory.components_gb,
            "Z-Image Encoder",
        ),
        dit=_generator_hardware_profiles(
            {1: 27.0187, 2: 16.7798, 4: 11.1702, 8: 8.7374},
            {1: 9659.144, 2: 6108.975, 4: 5498.181, 8: 4444.312},
            {
                "weights": {1: 11.46, 2: 11.46, 4: 11.46, 8: 11.46},
                "runtime": {1: 1.2905, 2: 0.8257, 4: 0.6655, 8: 0.6636},
            },
            "Z-Image DiT",
        ),
        vae=_generator_hardware_profiles(
            {1: 0.0122, 2: 0.0144, 4: 0.0191, 8: 0.0300},
            {1: 9.084, 2: 11.048, 4: 15.280, 8: 18.000},
            {
                "weights": {1: 0.31, 2: 0.31, 4: 0.31, 8: 0.31},
                "runtime": {1: 7.7163, 2: 7.5640, 4: 7.5386, 8: 7.5366},
            },
            "Z-Image VAE",
        ),
    )

    return TDProfileData(
        pe_models={"pe7b": pe7b},
        generator_models={model.name: model for model in (wan22, wan21, z_image)},
    )
