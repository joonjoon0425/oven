use candle_core::{Result, Tensor};
use candle_nn::VarMap;

use crate::traits::Batchable;

pub trait Encoder: Sized {
    type Obs: Batchable + Clone;
    type Config;

    fn new(vm: &VarMap, config: &Self::Config) -> Result<Self>;

    fn encode(&mut self, obs: &Self::Obs) -> Result<Tensor>;
    fn encode_batched(&mut self, batched_obs: &<Self::Obs as Batchable>::Batched) -> Result<Tensor>;
}

pub struct IdenEncoder;
impl Encoder for IdenEncoder {
    type Obs = Tensor;
    type Config = ();

    fn new(_vm: &VarMap, _config: &Self::Config) -> Result<Self> {
        Ok(IdenEncoder)
    }

    fn encode(&mut self, obs: &Self::Obs) -> Result<Tensor> {
        Ok(obs.clone())
    }

    fn encode_batched(&mut self, batched_obs: &<Self::Obs as Batchable>::Batched) -> Result<Tensor> {
        Ok(batched_obs.clone())
    }
}