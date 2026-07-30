use candle_core::{DType, Device};
use candle_nn::{Activation, BatchNorm, ParamsAdamW, linear, seq};
use oven::agent::{Agent, dqn::*};
use oven::components::buffer::replay_buffer::ReplayBuffer;
use oven::components::encoder::{Encoder, IdenEncoder};
use oven::components::mask::discretemask::DiscreteMask;
use oven::components::transition::Transition;
use oven::environment::cartpole::*;

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
        let mut obs = env.reset().unwrap();
        let mut steps = 0;
        let mut loss = 0f32;
        loop {
            let action = agent.explore(&obs, DiscreteMask::all_enabled(2)).unwrap();
            let (next_obs, reward, terminate, truncate) = env.step(action).unwrap();

            let transition = Transition {
                masked_observation: (obs, DiscreteMask::all_enabled(2)),
                action,
                reward,
                next_masked_observation: (next_obs.clone(), DiscreteMask::all_enabled(2)),
                terminate,
                truncate,
                extra: (),
            };

            buffer.push(transition);

            obs = next_obs;
            if let Some(batch) = &buffer.sample(64, &device).unwrap() {
                loss = agent.update(batch).unwrap();
            }
            steps += 1;
            counter += 1;
            
            if counter % 400 == 0 {
                agent.sync_networks().unwrap();
            }
            if terminate || truncate {
                break;
            }
            
        }
        let eps = agent.exploration().eps() * 0.99;
        *agent.exploration_mut().eps_mut() = if eps <= 0.05 { 0.05 } else { eps };
        if i % 10 == 0 {
            println!("EPISODE: {i:>5}, STEPS: {steps:>5}, LOSS: {loss:.5}, EPS: {:.5}, COUNTER: {}", agent.exploration().eps(), counter);
        }
    }
}