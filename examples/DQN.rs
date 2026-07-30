use candle_core::{DType, Device};
use candle_nn::{Activation, ParamsAdamW, linear, seq};
use oven::agent::{Agent, dqn::*};
use oven::components::buffer::replay_buffer::ReplayBuffer;
use oven::environment::cartpole::*;
use oven::episode::{EpisodeIterator};

fn main() {
    let seed = 12;
    let device = Device::new_cuda(0).unwrap();
    device.set_seed(seed).unwrap();
    
    let mut env = CartPoleEnv::new(&device, seed);
    let mut agent: DQNAgent<candle_nn::Sequential> = DQNAgent::builder(0.99, 2, seed)
        .network(&(), DType::F32, &device, &|vb| {
            let simple_net = seq()
                .add(linear(env.obs_dim(), 128, vb.pp("fc1"))?)
                .add(Activation::Relu)
                .add(linear(128, env.action_space_size(), vb.pp("fc2"))?);
                
            Ok(simple_net)
        }).unwrap()
        .exploration(1.0)
        .optimizer(ParamsAdamW {
            lr: 1e-3,
            ..Default::default()
        }).unwrap()
        .build().unwrap();
    
    let mut counter = 0;

    let mut buffer = ReplayBuffer::new(10000, seed);
    
    
    for i in 0..4000 {
        let mut loss = 0f32;
        let mut iter = DQNEpisode::new(&mut env, &mut agent, None);
        while let Some(t) = iter.step(&mut env, &mut agent).unwrap() {
            buffer.push(t);
            let batch = buffer.sample(32, &device).unwrap();
            if let Some(batch) = batch {
                loss = agent.update(&batch).unwrap();
            }
            counter += 1;
            if counter % 400 == 0 { agent.sync_networks().unwrap(); }
        }

        let eps = agent.exploration().eps() * 0.99;
        *agent.exploration_mut().eps_mut() = if eps <= 0.05 { 0.05 } else { eps };
        if i % 10 == 0 {
            println!("EPISODE: {i:>5}, STEPS: {:>5}, LOSS: {loss:.5}, EPS: {:.5}, COUNTER: {}", iter.steps(), agent.exploration().eps(), counter);
        }
    }
}