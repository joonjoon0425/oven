pub mod dqn;
use candle_core::Result;
use crate::episode::BasicEpisodeIterator;
use crate::{environment::Environment, traits::Batchable};
use crate::episode::EpisodeIterator;

pub trait Agent<Obs>: Sized
where
    Obs: Batchable + Clone
{
    type Action: Batchable + Clone;
    type Mask: Batchable + Clone;
    type Extra: Batchable + Clone;
    type Transition; // for convenience - just set it as Transition<Obs, Mask, Action, Extra>
    type BatchedTransition; // for convenience - just set is as BatchedTransition<Obs::Batched, Mask::Batched, Action::Batched, BatchedExtra>

    fn explore(&mut self, obs: &Obs, mask: Self::Mask) -> Result<Self::Action>;
    fn exploit(&mut self, obs: &Obs, mask: Self::Mask) -> Result<Self::Action>;
    fn update(&mut self, batch: &Self::BatchedTransition) -> Result<f32>;
}