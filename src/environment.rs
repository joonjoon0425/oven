use candle_core::Result;

use crate::traits::Batchable;

pub mod cartpole;
pub trait Environment {
    type Obs: Batchable + Clone;
    type Mask: Batchable + Clone;
    type Action: Batchable + Clone;

    fn step(&mut self, action: Self::Action) -> Result<(Self::Obs, Self::Mask, f32, bool, bool)>;
    fn reset(&mut self) -> Result<(Self::Obs, Self::Mask)>;
}