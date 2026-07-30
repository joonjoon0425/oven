use candle_nn::{Module, VarBuilder, VarMap};
use candle_core::{Tensor, Result, DType, Device};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{components::{encoder::{Encoder, IdenEncoder}, mask::discretemask::DiscreteMask}, traits::Batchable};

// Q Network
pub struct DiscreteQNet<M: Module, E: Encoder = IdenEncoder> {
    network: M,
    varmap: VarMap,
    encoder: E,
    rng: StdRng,
}

impl<M: Module, E: Encoder> DiscreteQNet<M, E> {
    pub fn new<F>(seed: u64, encoder_config: &E::Config, dtype: DType, device: &Device, func: &F) -> Result<DiscreteQNet<M, E>>
    where
        F: Fn(VarBuilder) -> Result<M>,
    {
        let varmap = VarMap::new();
        let varbuilder = VarBuilder::from_varmap(&varmap, dtype, device);
        let network = func(varbuilder)?;

        let encoder = E::new(&varmap, encoder_config)?;

        Ok(DiscreteQNet {
            network,
            varmap,
            encoder,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    pub fn varmap(&self) -> &VarMap { &self.varmap }
    pub fn varmap_mut(&mut self) -> &mut VarMap { &mut self.varmap }

    pub fn encoder(&self) -> &E { &self.encoder }

    pub fn max(&mut self, obs: &E::Obs, mask: DiscreteMask) -> Result<f32> {
        let mut max = f32::MIN;
        let qvalues = self.forward(obs)?.squeeze(0)?;
        for action in mask.iter() {
            let qvalue = qvalues.get(action as usize)?.to_scalar()?;
            if max <= qvalue {
                max = qvalue;
            }
        }
        Ok(max)
    }

    pub fn greedy_actions(&mut self, obs: &E::Obs, mask: DiscreteMask) -> Result<DiscreteMask> {
        let mut max = f32::MIN;
        let qvalues = self.forward(obs)?.squeeze(0)?;
        let mut greedy_actions = DiscreteMask::all_disabled(mask.n_actions);

        for action in mask.iter() {
            let qvalue = qvalues.get(action as usize)?.to_scalar()?;
            if max <= qvalue {
                max = qvalue;
                greedy_actions.disable_all();
                greedy_actions.enable(action as usize);
            } else if max - qvalue <= 1e-10 {
                greedy_actions.enable(action as usize);
            }
        }

        Ok(greedy_actions)
    }

    pub fn greedy_action(&mut self, obs: &E::Obs, mask: DiscreteMask) -> Result<u32> {
        let greedy_actions = self.greedy_actions(obs, mask)?;
        let n = greedy_actions.n_possible_actions();
        let r = self.rng.random_range(0..n);
        Ok(greedy_actions.iter().nth(r).unwrap())
    }

    pub fn forward(&mut self, obs: &E::Obs) -> Result<Tensor> {
        let encoded_obs = self.encoder.encode(obs)?.unsqueeze(0)?;
        self.network.forward(&encoded_obs)
    }

    pub fn batched_forward(&mut self, batched_obs: &<E::Obs as Batchable>::Batched) -> Result<Tensor> {
        let batched_encoded_obs = self.encoder.encode_batched(batched_obs)?;
        self.network.forward(&batched_encoded_obs)
    }

}

pub fn sync_varmap(src: &VarMap, dst: &mut VarMap) -> Result<()> {
    let src_data = src.data().lock().unwrap();
    let dst_data = dst.data().lock().unwrap();

    for (name, dst_var) in dst_data.iter() {
        match src_data.get(name) {
            Some(src_var) => {
                dst_var.set(&src_var.as_tensor().detach())?;
            },
            None => {
                panic!("The variable exists in target network, but does not exists in main network.")
            }
        }
    }
    Ok(())
}