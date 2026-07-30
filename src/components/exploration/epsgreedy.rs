use candle_core::{Result, Tensor};
use candle_nn::Module;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use crate::components::{discreteqnet::DiscreteQNet, encoder::Encoder, mask::discretemask::DiscreteMask};

pub struct EpsGreedy {
    eps: f32,
    rng: StdRng,
}

impl EpsGreedy {
    pub fn new(eps: f32, seed: u64) -> Self { Self{ eps, rng: StdRng::seed_from_u64(seed) } }

    pub fn sample<M: Module, E: Encoder>(&mut self, network: &mut DiscreteQNet<M, E>, obs: &E::Obs, mask: DiscreteMask) -> Result<u32> {
        let r = self.rng.random();

        if self.eps >= r {
            let n_possible_actions = mask.n_possible_actions();
            let n = self.rng.random_range(0..n_possible_actions);
            Ok(mask.iter().nth(n).unwrap())
        } else {
            Ok(network.greedy_action(obs, mask)?)
        }
    }

    pub fn eps(&self) -> f32 { self.eps }
    pub fn eps_mut(&mut self) -> &mut f32 { &mut self.eps }
}
