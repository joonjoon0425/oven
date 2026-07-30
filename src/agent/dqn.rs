use candle_core::{DType, Device, Result};
use candle_nn::{AdamW, Module, Optimizer, VarBuilder, VarMap, loss};
use rand::{RngExt, SeedableRng};
use rand::rngs::StdRng;

use crate::agent::Agent;
use crate::components::encoder::{Encoder, IdenEncoder};
use crate::components::exploration::epsgreedy::EpsGreedy;
use crate::components::mask::discretemask::DiscreteMask;
use crate::components::discreteqnet::*;
use crate::components::transition::{BatchedTransition, Transition};
use crate::traits::Batchable;

pub struct DQNAgent<M: Module, E: Encoder = IdenEncoder, Opt: Optimizer = AdamW> {
    gamma: f32,
    n_actions: usize,

    main_network: DiscreteQNet<M, E>,
    target_network: DiscreteQNet<M, E>,

    exploration: EpsGreedy,

    rng: StdRng,

    optimizer: Opt,
}

pub struct DQNAgentBuilder<M: Module, E: Encoder = IdenEncoder, Opt: Optimizer = AdamW> {
    gamma: f32,
    explore: Option<EpsGreedy>,
    n_actions: usize,
    rng: StdRng,
    main_network: Option<DiscreteQNet<M, E>>,
    target_network: Option<DiscreteQNet<M, E>>,
    encoder: Option<E>,
    optimizer: Option<Opt>
}

impl<M: Module, E: Encoder, Opt: Optimizer> Agent<E::Obs> for DQNAgent<M, E, Opt> {
    type Action = u32;
    type Mask = DiscreteMask;
    type Extra = ();
    type Transition = Transition<E::Obs, Self::Action, Self::Mask, Self::Extra>;
    type BatchedTransition = BatchedTransition<<E::Obs as Batchable>::Batched, <u32 as Batchable>::Batched, <Self::Mask as Batchable>::Batched, ()>;
    
    fn explore(&mut self, obs: &E::Obs, mask: Self::Mask) -> Result<Self::Action> {
        self.exploration.sample(&mut self.main_network, obs, mask)
    }

    fn exploit(&mut self, obs: &E::Obs, mask: Self::Mask) -> Result<Self::Action> {
        self.target_network.greedy_action(&mut self.rng, obs, mask)
    }

    fn update(&mut self, batch: &Self::BatchedTransition) -> Result<f32> {
        let batched_obs = &batch.observations;
        let batched_qvalues = self.main_network.batched_forward(&batched_obs)?;
        let predicted_qvalues = batched_qvalues.gather(&batch.actions.unsqueeze(1)?, 1)?.squeeze(1)?;

        let next_batched_obs = &batch.next_observations;
        let batched_targets = self.target_network.batched_forward(&next_batched_obs)?.detach();
        let bootstrap = (batched_targets.max(1)? * batch.terminates.affine(-1., 1.))?;
        let target_qvalues = (&batch.rewards + bootstrap.affine(self.gamma as f64, 0f64))?.detach();

        // use huber loss for now
        let loss = loss::huber(&predicted_qvalues, &target_qvalues, 2.)?;
        self.optimizer.backward_step(&loss)?;

        Ok(loss.to_scalar()?)
    }
}

impl<M: Module, E: Encoder, Opt: Optimizer> DQNAgent<M, E, Opt> {
    pub fn builder(gamma: f32, n_actions: usize, seed: u64) -> DQNAgentBuilder<M, E, Opt> {
        DQNAgentBuilder {
            gamma,
            n_actions,
            rng: StdRng::seed_from_u64(seed),
            explore: None,
            target_network: None,
            main_network: None,
            encoder: None,
            optimizer: None,
        }
    }

    pub fn sync_networks(&mut self) -> Result<()> {
        sync_varmap(self.main_network.varmap(), self.target_network.varmap_mut())
    }

    pub fn exploration(&self) -> &EpsGreedy {
        &self.exploration
    }

    pub fn exploration_mut(&mut self) -> &mut EpsGreedy {
        &mut self.exploration
    }
}

impl<M: Module, E: Encoder, Opt: Optimizer> DQNAgentBuilder<M, E, Opt>{
    pub fn exploration(mut self, eps: f32) -> Self {
        self.explore = EpsGreedy::new(eps, self.rng.random()).into();
        self
    }

    pub fn optimizer(mut self, config: Opt::Config) -> Result<Self> {
        let main_network = self.main_network.unwrap();
        self.optimizer = Opt::new(main_network.varmap().all_vars(), config)?.into();
        self.main_network = main_network.into();
        Ok(self)
    }

    pub fn network<F>(mut self, config: &E::Config, dtype: DType, device: &Device, func: &F) -> Result<Self>
    where
        F: Fn(VarBuilder) -> Result<M>,
    {
        let main_network = DiscreteQNet::new(self.rng.random(), config, dtype, device, func)?;
        let target_network = DiscreteQNet::new(self.rng.random(), config, dtype, device, func)?;

        self.main_network = main_network.into();
        self.target_network = target_network.into();

        Ok(self)
    }

    pub fn build(self) -> Result<DQNAgent<M, E, Opt>> {
        Ok(DQNAgent {
            gamma: self.gamma,
            n_actions: self.n_actions,
            main_network: self.main_network.unwrap(),
            target_network: self.target_network.unwrap(),
            exploration: self.explore.unwrap(),
            rng: self.rng,
            optimizer: self.optimizer.unwrap(),
        })
    }
}