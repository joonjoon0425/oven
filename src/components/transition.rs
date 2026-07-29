use crate::traits::Batchable;
use candle_core::Tensor;
pub struct Transition<Obs, Action, Mask = (), Extra = ()>
where
    Obs: Batchable + Clone,
    Mask: Batchable + Clone,
    Action: Batchable + Clone,
    Extra: Batchable + Clone
{
    pub masked_observation: (Obs, Mask),
    pub action: Action,
    pub reward: f32,
    pub next_masked_observation: (Obs, Mask),
    pub terminate: bool,
    pub truncate: bool,
    pub extra: Extra, 
}

pub struct BatchedTransition<BatchedObs, BatchedAction, BatchedMask, BatchedExtra = ()> {
    pub observations: BatchedObs,
    pub masks: BatchedMask,
    pub actions: BatchedAction,
    pub rewards: Tensor,
    pub next_observations: BatchedObs,
    pub next_masks: BatchedMask,
    pub terminates: Tensor,
    pub truncates: Tensor,
    pub extras: BatchedExtra,
}