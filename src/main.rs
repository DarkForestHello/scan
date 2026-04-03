use std::borrow::Cow;
use std::num::NonZeroU64;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use wgpu::util::DeviceExt;

const INPUT_LEN: usize = 262_144;
const INPUT_SEED: u64 = 0x5EED_1234_5678_9ABC;
const WARMUP_ITERATIONS: u32 = 100;
const BENCH_ITERATIONS: u32 = 2_000;
const WORKGROUP_SIZES: &[u32] = &[64, 128, 256, 512, 1024];
const ELEMENTS_PER_THREAD_OPTIONS: &[u32] = &[2, 4, 8];

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ScanParams {
    total_len: u32,
    num_blocks: u32,
    block_len: u32,
    epoch: u32,
}

#[derive(Clone, Copy, Debug)]
struct ScanConfig {
    workgroup_size: u32,
    elements_per_thread: u32,
}

impl ScanConfig {
    const fn new(workgroup_size: u32, elements_per_thread: u32) -> Self {
        Self {
            workgroup_size,
            elements_per_thread,
        }
    }

    fn elements_per_block(self) -> u32 {
        self.workgroup_size * self.elements_per_thread
    }
}

#[derive(Debug)]
struct BenchmarkResult {
    config: ScanConfig,
    output: Vec<i32>,
    gpu_total_time_ms: f64,
    gpu_kernel_time_ms: f64,
    validation_ok: bool,
    used_subgroup: bool,
    num_blocks: u32,
    elements_per_block: u32,
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    supports_timestamps: bool,
    limits: wgpu::Limits,
    supports_subgroup: bool,
}

struct ScanRuntime {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    output_buffer: wgpu::Buffer,
    state_buffer: wgpu::Buffer,
    aggregate_buffer: wgpu::Buffer,
    prefix_buffer: wgpu::Buffer,
    num_blocks: u32,
    elements_per_block: u32,
    param_stride: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let input = generate_input();
    let cpu_output = cpu_inclusive_scan(&input);
    let cpu_time_ms = benchmark_cpu(&input, BENCH_ITERATIONS);

    let gpu = create_gpu_context().await?;
    let configs = supported_configs(&gpu);
    let mut results = Vec::with_capacity(configs.len());
    for config in configs {
        let result = benchmark_gpu_config(&gpu, config, &input, &cpu_output).await?;
        results.push(result);
    }

    let best = results
        .iter()
        .filter(|result| result.validation_ok)
        .min_by(|a, b| a.gpu_kernel_time_ms.total_cmp(&b.gpu_kernel_time_ms))
        .ok_or("No GPU configuration passed validation")?;

    println!("Algorithm              : single-pass decoupled lookback");
    println!("CPU benchmark iterations: {}", BENCH_ITERATIONS);
    println!("GPU warmup iterations  : {}", WARMUP_ITERATIONS);
    println!("GPU benchmark iterations: {}", BENCH_ITERATIONS);
    println!(
        "Device max workgroup size: {}",
        gpu.limits.max_compute_workgroup_size_x
    );
    println!(
        "Subgroup support        : {}",
        if gpu.supports_subgroup {
            "enabled"
        } else {
            "disabled"
        }
    );
    if gpu.supports_subgroup {
        println!(
            "Subgroup size range     : {}..={}",
            gpu.limits.min_subgroup_size, gpu.limits.max_subgroup_size
        );
    }
    println!("CPU avg time           : {:.6} ms", cpu_time_ms);
    println!(
        "Best config            : wg={}, ept={}, subgroup={}",
        best.config.workgroup_size,
        best.config.elements_per_thread,
        if best.used_subgroup { "on" } else { "off" }
    );
    println!("Best num blocks        : {}", best.num_blocks);
    println!("Elements per block     : {}", best.elements_per_block);
    println!(
        "Lookback max depth     : {}",
        best.num_blocks.saturating_sub(1)
    );
    println!(
        "Validation             : {}",
        if best.validation_ok { "PASS" } else { "FAIL" }
    );
    println!("Input first 50         : {:?}", &input[..50]);
    println!("Input last 50          : {:?}", &input[input.len() - 50..]);
    println!("Output first 50        : {:?}", &best.output[..50]);
    println!(
        "Output last 50         : {:?}",
        &best.output[best.output.len() - 50..]
    );
    println!("Best GPU total time    : {:.6} ms", best.gpu_total_time_ms);
    println!("Best GPU kernel time   : {:.6} ms", best.gpu_kernel_time_ms);
    println!(
        "Metadata/reset overhead: {:.6} ms",
        (best.gpu_total_time_ms - best.gpu_kernel_time_ms).max(0.0)
    );
    println!(
        "Kernel GPU/CPU speedup : {:.3}x",
        if best.gpu_kernel_time_ms == 0.0 {
            f64::INFINITY
        } else {
            cpu_time_ms / best.gpu_kernel_time_ms
        }
    );
    println!("All configs:");
    for result in &results {
        println!(
            "  wg={}, ept={}, subgroup={}, blocks={}, elems/block={} -> total {:.6} ms, kernel {:.6} ms, {}",
            result.config.workgroup_size,
            result.config.elements_per_thread,
            if result.used_subgroup { "on" } else { "off" },
            result.num_blocks,
            result.elements_per_block,
            result.gpu_total_time_ms,
            result.gpu_kernel_time_ms,
            if result.validation_ok { "PASS" } else { "FAIL" }
        );
    }

    Ok(())
}

fn generate_input() -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(INPUT_SEED);
    (0..INPUT_LEN).map(|_| rng.gen_range(0..=9)).collect()
}

fn cpu_inclusive_scan(input: &[i32]) -> Vec<i32> {
    let mut output = Vec::with_capacity(input.len());
    let mut acc = 0_i32;
    for &value in input {
        acc += value;
        output.push(acc);
    }
    output
}

fn benchmark_cpu(input: &[i32], iterations: u32) -> f64 {
    let start = Instant::now();
    let mut last_output = Vec::new();
    for _ in 0..iterations {
        last_output = cpu_inclusive_scan(input);
    }
    std::hint::black_box(&last_output);
    start.elapsed().as_secs_f64() * 1_000.0 / f64::from(iterations)
}

fn supported_configs(gpu: &GpuContext) -> Vec<ScanConfig> {
    WORKGROUP_SIZES
        .iter()
        .copied()
        .filter(|&workgroup_size| {
            workgroup_size <= gpu.limits.max_compute_workgroup_size_x
                && workgroup_size <= gpu.limits.max_compute_invocations_per_workgroup
        })
        .flat_map(|workgroup_size| {
            ELEMENTS_PER_THREAD_OPTIONS
                .iter()
                .copied()
                .map(move |elements_per_thread| {
                    ScanConfig::new(workgroup_size, elements_per_thread)
                })
        })
        .collect()
}

async fn create_gpu_context() -> Result<GpuContext, Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or("No Vulkan adapter available")?;

    let adapter_features = adapter.features();
    let adapter_limits = adapter.limits();
    let mut required_features = wgpu::Features::empty();
    if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
        required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    }
    let supports_subgroup = adapter_features.contains(wgpu::Features::SUBGROUP);
    if supports_subgroup {
        required_features |= wgpu::Features::SUBGROUP;
    }
    if adapter_features.contains(wgpu::Features::SUBGROUP_BARRIER) {
        required_features |= wgpu::Features::SUBGROUP_BARRIER;
    }

    let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits.clone());
    limits.max_storage_buffers_per_shader_stage =
        adapter_limits.max_storage_buffers_per_shader_stage;
    limits.max_compute_workgroup_size_x = adapter_limits.max_compute_workgroup_size_x;
    limits.max_compute_invocations_per_workgroup =
        adapter_limits.max_compute_invocations_per_workgroup;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("scan-device"),
                required_features,
                required_limits: limits.clone(),
            },
            None,
        )
        .await?;

    let supports_timestamps = required_features.contains(wgpu::Features::TIMESTAMP_QUERY)
        && required_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

    Ok(GpuContext {
        device,
        queue,
        supports_timestamps,
        limits,
        supports_subgroup,
    })
}

async fn benchmark_gpu_config(
    gpu: &GpuContext,
    config: ScanConfig,
    input: &[i32],
    expected: &[i32],
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let runtime = create_runtime(&gpu.device, config, input, gpu.supports_subgroup);

    let validation_readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("validation-readback"),
        size: runtime.output_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    run_scan_dispatches(
        gpu,
        &runtime,
        1,
        Some((&runtime.output_buffer, &validation_readback)),
        false,
    )
    .await?;
    let output = read_buffer::<i32>(&gpu.device, &validation_readback).await?;
    let validation_ok = output == expected;

    run_scan_dispatches(gpu, &runtime, WARMUP_ITERATIONS, None, true).await?;

    let total_gpu_time_ms = run_scan_dispatches(
        gpu,
        &runtime,
        BENCH_ITERATIONS,
        None,
        true,
    )
    .await?;

    let kernel_gpu_time_ms =
        run_scan_dispatches(gpu, &runtime, BENCH_ITERATIONS, None, false).await?;

    Ok(BenchmarkResult {
        config,
        output,
        gpu_total_time_ms: total_gpu_time_ms / f64::from(BENCH_ITERATIONS),
        gpu_kernel_time_ms: kernel_gpu_time_ms / f64::from(BENCH_ITERATIONS),
        validation_ok,
        used_subgroup: gpu.supports_subgroup,
        num_blocks: runtime.num_blocks,
        elements_per_block: runtime.elements_per_block,
    })
}

fn create_runtime(
    device: &wgpu::Device,
    config: ScanConfig,
    input: &[i32],
    use_subgroup: bool,
) -> ScanRuntime {
    let shader_source = generate_shader_source(config, use_subgroup);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scan-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source)),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scan-bind-group-layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, false),
            storage_entry(3, false),
            storage_entry(4, false),
            uniform_entry(5, true),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("scan-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = create_pipeline(device, &pipeline_layout, &shader, "decoupled_lookback_scan");
    let num_blocks = div_ceil(input.len() as u32, config.elements_per_block());

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input-buffer"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output-buffer"),
        size: (input.len() * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let metadata_size = (num_blocks as u64) * std::mem::size_of::<u32>() as u64;
    let state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("state-buffer"),
        size: metadata_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let aggregate_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aggregate-buffer"),
        size: metadata_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let prefix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prefix-buffer"),
        size: metadata_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let param_stride = device
        .limits()
        .min_uniform_buffer_offset_alignment
        .max(std::mem::size_of::<ScanParams>() as u32);
    let max_iterations = WARMUP_ITERATIONS.max(BENCH_ITERATIONS);
    let mut params_bytes = vec![0_u8; param_stride as usize * max_iterations as usize];
    for epoch in 0..max_iterations {
        let params = ScanParams {
            total_len: input.len() as u32,
            num_blocks,
            block_len: config.elements_per_block(),
            epoch: epoch + 1,
        };
        let start = epoch as usize * param_stride as usize;
        let end = start + std::mem::size_of::<ScanParams>();
        params_bytes[start..end].copy_from_slice(bytemuck::bytes_of(&params));
    }
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params-buffer"),
        contents: &params_bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scan-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: aggregate_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: prefix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params_buffer,
                    offset: 0,
                    size: NonZeroU64::new(std::mem::size_of::<ScanParams>() as u64),
                }),
            },
        ],
    });

    ScanRuntime {
        pipeline,
        bind_group,
        output_buffer,
        state_buffer,
        aggregate_buffer,
        prefix_buffer,
        num_blocks,
        elements_per_block: config.elements_per_block(),
        param_stride,
    }
}

async fn run_scan_dispatches(
    gpu: &GpuContext,
    runtime: &ScanRuntime,
    iterations: u32,
    readback: Option<(&wgpu::Buffer, &wgpu::Buffer)>,
    include_reset_overhead: bool,
) -> Result<f64, Box<dyn std::error::Error>> {
    let (query_set, query_resolve_buffer, query_readback_buffer) = if gpu.supports_timestamps {
        let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamp-query-set"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let query_resolve_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp-query-resolve-buffer"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });
        let query_readback_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp-query-readback-buffer"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        (
            Some(query_set),
            Some(query_resolve_buffer),
            Some(query_readback_buffer),
        )
    } else {
        (None, None, None)
    };

    let fallback_timer = Instant::now();
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scan-encoder"),
        });

    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 0);
    }

    for iter in 0..iterations {
        if include_reset_overhead {
            encoder.clear_buffer(&runtime.state_buffer, 0, None);
            encoder.clear_buffer(&runtime.aggregate_buffer, 0, None);
            encoder.clear_buffer(&runtime.prefix_buffer, 0, None);
        }
        let pass_label = if include_reset_overhead {
            "scan-pass-total"
        } else {
            "scan-pass-kernel"
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass_label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.pipeline);
        let dynamic_offset = runtime.param_stride * iter;
        pass.set_bind_group(0, &runtime.bind_group, &[dynamic_offset]);
        pass.dispatch_workgroups(runtime.num_blocks, 1, 1);
    }

    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
        encoder.resolve_query_set(query_set, 0..2, query_resolve_buffer.as_ref().unwrap(), 0);
        encoder.copy_buffer_to_buffer(
            query_resolve_buffer.as_ref().unwrap(),
            0,
            query_readback_buffer.as_ref().unwrap(),
            0,
            16,
        );
    }

    if let Some((src, dst)) = readback {
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, src.size());
    }

    gpu.queue.submit(Some(encoder.finish()));
    gpu.device.poll(wgpu::Maintain::Wait);

    let elapsed_ms = if let Some(query_buffer) = query_readback_buffer.as_ref() {
        let timestamps = read_buffer::<u64>(&gpu.device, query_buffer).await?;
        let period = f64::from(gpu.queue.get_timestamp_period());
        if timestamps.len() == 2 && timestamps[1] >= timestamps[0] {
            (timestamps[1] - timestamps[0]) as f64 * period / 1_000_000.0
        } else {
            fallback_timer.elapsed().as_secs_f64() * 1_000.0
        }
    } else {
        fallback_timer.elapsed().as_secs_f64() * 1_000.0
    };

    Ok(elapsed_ms)
}

fn generate_shader_source(config: ScanConfig, use_subgroup: bool) -> String {
    include_str!("scan.wgsl")
        .replace("{{WORKGROUP_SIZE}}", &config.workgroup_size.to_string())
        .replace(
            "{{ELEMENTS_PER_THREAD}}",
            &config.elements_per_thread.to_string(),
        )
        .replace(
            "{{USE_SUBGROUP}}",
            if use_subgroup { "true" } else { "false" },
        )
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(layout),
        module: shader,
        entry_point,
        compilation_options: Default::default(),
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, has_dynamic_offset: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset,
            min_binding_size: NonZeroU64::new(std::mem::size_of::<ScanParams>() as u64),
        },
        count: None,
    }
}

async fn read_buffer<T: Pod>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()?
        .map_err(|e| format!("buffer map failed: {e}"))?;

    let data = slice.get_mapped_range();
    let output = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buffer.unmap();
    Ok(output)
}

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}
