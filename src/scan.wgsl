struct ScanParams {
    total_len: u32,
    num_blocks: u32,
    block_len: u32,
    epoch: u32,
}

const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;
const ELEMENTS_PER_THREAD: u32 = {{ELEMENTS_PER_THREAD}}u;
const USE_SUBGROUP: bool = {{USE_SUBGROUP}};
const BLOCK_STATE_EMPTY: u32 = 0u;
const BLOCK_STATE_AGGREGATE: u32 = 1u;
const BLOCK_STATE_PREFIX: u32 = 2u;
const STATE_BITS: u32 = 2u;

@group(0) @binding(0)
var<storage, read> input_data: array<i32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<i32>;

@group(0) @binding(2)
var<storage, read_write> block_states: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read_write> block_aggregates: array<atomic<i32>>;

@group(0) @binding(4)
var<storage, read_write> block_prefixes: array<atomic<i32>>;

@group(0) @binding(5)
var<uniform> params: ScanParams;

var<workgroup> shared_data: array<i32, {{WORKGROUP_SIZE}}>;
var<workgroup> shared_block_total: i32;
var<workgroup> shared_block_prefix: i32;

fn load_value(index: u32) -> i32 {
    if (index < params.total_len) {
        return input_data[index];
    }
    return 0;
}

fn pack_state(epoch: u32, state: u32) -> u32 {
    return (epoch << STATE_BITS) | state;
}

fn load_state_for_epoch(block_index: u32, epoch: u32) -> u32 {
    let raw = atomicLoad(&block_states[block_index]);
    if ((raw >> STATE_BITS) != epoch) {
        return BLOCK_STATE_EMPTY;
    }
    return raw & ((1u << STATE_BITS) - 1u);
}

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn decoupled_lookback_scan(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32
) {
    let lid = local_id.x;
    let block_index = workgroup_id.x;
    let block_base = block_index * params.block_len;
    let thread_base = block_base + lid * ELEMENTS_PER_THREAD;

    var thread_prefixes: array<i32, {{ELEMENTS_PER_THREAD}}>;
    var running_sum = 0i;

    var i = 0u;
    loop {
        if (i >= ELEMENTS_PER_THREAD) {
            break;
        }

        let index = thread_base + i;
        running_sum = running_sum + load_value(index);
        thread_prefixes[i] = running_sum;
        i = i + 1u;
    }

    shared_data[lid] = running_sum;
    workgroupBarrier();
    var thread_exclusive = 0i;
    if (USE_SUBGROUP) {
        let subgroup_inclusive = subgroupInclusiveAdd(running_sum);
        let subgroup_exclusive = subgroup_inclusive - running_sum;
        if (subgroup_invocation_id + 1u == subgroup_size) {
            shared_data[subgroup_id] = subgroup_inclusive;
        }
        workgroupBarrier();

        if (lid == 0u) {
            var acc = 0i;
            var sg = 0u;
            loop {
                if (sg >= num_subgroups) {
                    break;
                }
                let subtotal = shared_data[sg];
                shared_data[sg] = acc;
                acc = acc + subtotal;
                sg = sg + 1u;
            }
            shared_block_total = acc;
        }
        workgroupBarrier();
        thread_exclusive = shared_data[subgroup_id] + subgroup_exclusive;
    } else {
        var stride = 1u;
        loop {
            if (stride >= WORKGROUP_SIZE) {
                break;
            }

            let bi = (lid + 1u) * stride * 2u - 1u;
            if (bi < WORKGROUP_SIZE) {
                shared_data[bi] = shared_data[bi] + shared_data[bi - stride];
            }
            workgroupBarrier();
            stride = stride * 2u;
        }

        if (lid == 0u) {
            shared_block_total = shared_data[WORKGROUP_SIZE - 1u];
            shared_data[WORKGROUP_SIZE - 1u] = 0;
        }
        workgroupBarrier();

        stride = WORKGROUP_SIZE / 2u;
        loop {
            let bi = (lid + 1u) * stride * 2u - 1u;
            if (bi < WORKGROUP_SIZE) {
                let ai = bi - stride;
                let left = shared_data[ai];
                let right = shared_data[bi];
                shared_data[ai] = right;
                shared_data[bi] = right + left;
            }

            workgroupBarrier();
            if (stride == 1u) {
                break;
            }
            stride = stride / 2u;
        }
        thread_exclusive = shared_data[lid];
    }

    if (lid == 0u) {
        atomicStore(&block_aggregates[block_index], shared_block_total);
        if (block_index != 0u) {
            atomicStore(
                &block_states[block_index],
                pack_state(params.epoch, BLOCK_STATE_AGGREGATE),
            );
        }
    }
    workgroupBarrier();

    if (lid == 0u) {
        var block_prefix = 0i;
        if (block_index != 0u) {
            var lookback = i32(block_index) - 1;
            loop {
                let candidate = u32(lookback);
                let state = load_state_for_epoch(candidate, params.epoch);
                if (state == BLOCK_STATE_EMPTY) {
                    continue;
                }
                if (state == BLOCK_STATE_PREFIX) {
                    block_prefix = block_prefix + atomicLoad(&block_prefixes[candidate]);
                    break;
                }

                block_prefix = block_prefix + atomicLoad(&block_aggregates[candidate]);
                if (lookback == 0) {
                    break;
                }
                lookback = lookback - 1;
            }
        }

        shared_block_prefix = block_prefix;
        atomicStore(
            &block_prefixes[block_index],
            block_prefix + shared_block_total,
        );
        atomicStore(
            &block_states[block_index],
            pack_state(params.epoch, BLOCK_STATE_PREFIX),
        );
    }
    workgroupBarrier();

    let global_thread_offset = shared_block_prefix + thread_exclusive;
    i = 0u;
    loop {
        if (i >= ELEMENTS_PER_THREAD) {
            break;
        }

        let index = thread_base + i;
        if (index < params.total_len) {
            output_data[index] = global_thread_offset + thread_prefixes[i];
        }
        i = i + 1u;
    }
}
