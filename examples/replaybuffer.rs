use candle_core::{Device, Result, Tensor};
use oven::components::buffer::replay_buffer::*;
use oven::components::transition::*;
use rand::{random, random_range};

fn dummy_transition(counter:u32, device: &Device) -> Result<Transition<Tensor, u32>> {
    Ok(Transition {
        masked_observation: (Tensor::randn(0.0, 1.0, (2, 3), device)?, ()),
        action: counter,
        reward: 2.0,
        next_masked_observation: (Tensor::randn(0.0, 1.0, (2, 3), device)?, ()),
        terminate: false,
        truncate: false,
        extra: ()
    })
}

fn main() {
    let device = Device::Cpu;
    let seed = 10;

    let mut buf = ReplayBuffer::new(30, seed);

    for i in 0..50 {
        buf.push(dummy_transition(i, &device).unwrap())
    }

    for (i, (observation, action)) in buf.observations().iter().zip(buf.actions()).enumerate() {
        println!("obs {i}: {}", observation);
        println!("action {i}: {}", action);
    }

    for _ in 0..10 {
        let batched = buf.sample(16, &device).unwrap().unwrap();
        println!("obs: {}", batched.observations);
        println!("action: {}", batched.actions);
    }
}