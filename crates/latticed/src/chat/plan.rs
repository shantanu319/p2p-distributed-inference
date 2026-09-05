use anyhow::{Result, bail};
use lattice_engine::ModelInventory;
use lattice_net::GpuDevice;

pub struct Plan {
    pub layers: Vec<u32>,
}

pub fn build(
    inventory: &ModelInventory,
    context: u32,
    devices: &[GpuDevice],
    reserve: u64,
) -> Result<Plan> {
    let facts = &inventory.facts;
    if facts.layers == 0
        || facts.layers > 4096
        || facts.hidden_dim == 0
        || facts.hidden_dim.checked_rem(facts.heads) != Some(0)
        || facts.kv_heads == 0
        || facts.kv_heads > facts.heads
        || inventory.layer_bytes.len() != facts.layers as usize
    {
        bail!("unsupported model dimensions for the Llama memory planner");
    }
    let count = devices.len();
    let layers = facts.layers as usize;
    if count == 0 || count > 63 || count > layers {
        bail!(
            "select between 1 and {} GPUs so each receives transformer layers",
            layers.min(63)
        );
    }
    let budgets: Vec<u128> = devices
        .iter()
        .map(|device| {
            let free = device.free_memory.min(device.total_memory);
            u128::from(free.saturating_sub(reserve.max(free / 10)))
        })
        .collect();
    let kv = 4
        * u128::from(facts.kv_heads)
        * u128::from(facts.hidden_dim / facts.heads)
        * u128::from(context);
    let mut prefix = vec![0u128; layers + 1];
    for (index, bytes) in inventory.layer_bytes.iter().enumerate() {
        prefix[index + 1] = prefix[index] + u128::from(*bytes) + kv;
    }
    let output = u128::from(inventory.output_bytes);
    let mut feasible = vec![vec![false; layers + 1]; count + 1];
    feasible[count][layers] = true;
    for device in (0..count).rev() {
        let extra = if device == count - 1 { output } else { 0 };
        let Some(budget) = budgets[device].checked_sub(extra) else {
            continue;
        };
        let mut reachable = vec![0usize; layers + 2];
        for end in 0..=layers {
            reachable[end + 1] = reachable[end] + usize::from(feasible[device + 1][end]);
        }
        let mut end = 0;
        for start in 0..layers {
            end = end.max(start);
            while end < layers && prefix[end + 1] - prefix[start] <= budget {
                end += 1;
            }
            feasible[device][start] = reachable[end + 1] > reachable[start + 1];
        }
    }
    if !feasible[0][0] {
        bail!(
            "no contiguous layer allocation fits the selected GPUs after memory reserves, including f16 KV and output tensors; reduce --context, use a smaller quantization, or change workers"
        );
    }
    let total_budget: u128 = budgets.iter().sum();
    let total_bytes = prefix[layers] + output;
    let mut assigned = Vec::with_capacity(count);
    let mut start = 0;
    let mut cumulative_budget = 0;
    for device in 0..count {
        cumulative_budget += budgets[device];
        let extra = if device == count - 1 { output } else { 0 };
        let target = total_bytes as f64 * cumulative_budget as f64 / total_budget as f64;
        let end = (start + 1..=layers)
            .take_while(|&end| prefix[end] - prefix[start] + extra <= budgets[device])
            .filter(|&end| feasible[device + 1][end])
            .min_by(|&a, &b| {
                let distance = |end: usize| ((prefix[end] + extra) as f64 - target).abs();
                distance(a).total_cmp(&distance(b))
            })
            .expect("a feasible partition has a next cut");
        assigned.push((end - start) as u32);
        start = end;
    }
    *assigned.last_mut().unwrap() += 1;
    Ok(Plan { layers: assigned })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_engine::ModelFacts;

    fn inventory(weights: &[u64], output_bytes: u64) -> ModelInventory {
        ModelInventory {
            facts: ModelFacts {
                arch: "llama".into(),
                layers: weights.len() as u32,
                hidden_dim: 1,
                heads: 1,
                kv_heads: 1,
                train_context: 2048,
            },
            layer_bytes: weights.to_vec(),
            output_bytes,
        }
    }

    fn device(bytes: u64) -> GpuDevice {
        GpuDevice {
            name: "GPU".into(),
            description: "test GPU".into(),
            kind: "gpu".into(),
            total_memory: bytes,
            free_memory: bytes,
        }
    }

    #[test]
    fn uneven_layers_fit_without_averaging_their_sizes() {
        let plan = build(
            &inventory(&[800, 100, 100], 0),
            0,
            &[device(900), device(300)],
            0,
        )
        .unwrap();
        assert_eq!(plan.layers, vec![1, 3]);
    }

    #[test]
    fn large_output_is_charged_to_the_final_gpu() {
        let model = inventory(&[100, 100, 100, 100], 700);
        let plan = build(&model, 0, &[device(400), device(900)], 0).unwrap();
        assert_eq!(plan.layers, vec![3, 2]);
        assert!(build(&model, 0, &[device(900), device(400)], 0).is_err());
    }

    #[test]
    fn every_gpu_gets_transformer_layers_and_context_is_charged() {
        let model = inventory(&[100; 6], 50);
        let plan = build(&model, 0, &[device(300), device(900)], 0).unwrap();
        assert_eq!(plan.layers.iter().sum::<u32>(), 7);
        assert!(plan.layers[0] >= 1 && plan.layers[1] >= 2);
        assert!(build(&model, 100, &[device(300), device(900)], 0).is_err());
    }

    #[test]
    fn free_memory_and_reserves_limit_admission() {
        let mut gpu = device(1000);
        gpu.free_memory = 200;
        assert!(build(&inventory(&[100], 100), 0, &[gpu], 0).is_err());
        assert!(build(&inventory(&[100], 100), 0, &[device(1000)], 900).is_err());
    }
    #[test]
    fn feasibility_matches_exhaustive_three_gpu_partitions() {
        let model = inventory(&[70, 180, 40, 120, 90], 80);
        for a in [100, 250, 500] {
            for b in [100, 250, 500] {
                for c in [100, 250, 500] {
                    let capacities = [a, b, c].map(|n| n - n / 10);
                    let possible = (1..4).any(|x| {
                        (x + 1..5).any(|y| {
                            model.layer_bytes[..x].iter().sum::<u64>() <= capacities[0]
                                && model.layer_bytes[x..y].iter().sum::<u64>() <= capacities[1]
                                && model.layer_bytes[y..].iter().sum::<u64>() + model.output_bytes
                                    <= capacities[2]
                        })
                    });
                    assert_eq!(
                        build(&model, 0, &[device(a), device(b), device(c)], 0).is_ok(),
                        possible
                    );
                }
            }
        }
    }
}
