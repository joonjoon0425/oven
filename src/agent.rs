pub mod dqn;
use candle_core::Result;
use crate::traits::{Batchable};

pub trait Agent<Obs>
where
    Obs: Batchable
{
    type Action: Batchable;
    type Mask: Batchable;
    type Extra: Batchable;
    type Transition; // for convenience - just set it as Transition<Obs, Mask, Action, Extra>
    type BatchedTransition; // for convenience - just set is as BatchedTransition<Obs::Batched, Mask::Batched, Action::Batched, BatchedExtra>

    fn explore(&mut self, obs: &Obs, mask: Self::Mask) -> Result<Self::Action>;
    fn exploit(&mut self, obs: &Obs, mask: Self::Mask) -> Result<Self::Action>;

    fn update(&mut self, batch: &Self::BatchedTransition) -> Result<f32>;
}