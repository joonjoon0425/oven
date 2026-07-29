use candle_nn::{Module, VarBuilder, VarMap};
use candle_core::{Tensor, Result, DType, Device};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::components::{mask::discretemask::DiscreteMask};

// Q Network
pub struct DiscreteQNet<M: Module> {
    network: M,
    varmap: VarMap,
    rng: StdRng,
}

impl<M: Module> DiscreteQNet<M> {
    pub fn new<F>(seed: u64, dtype: DType, device: &Device, func: &F) -> Result<DiscreteQNet<M>>
    where
        F: Fn(VarBuilder) -> Result<M>,
    {
        let varmap = VarMap::new();
        let varbuilder = VarBuilder::from_varmap(&varmap, dtype, device);

        let network = func(varbuilder)?;

        Ok(DiscreteQNet {
            network,
            varmap,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    pub fn varmap(&self) -> &VarMap { &self.varmap }
    pub fn varmap_mut(&mut self) -> &mut VarMap { &mut self.varmap }

    pub fn max(&mut self, obs: &Tensor, mask: DiscreteMask) -> Result<f32> {
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

    pub fn greedy_actions(&mut self, obs: &Tensor, mask: DiscreteMask) -> Result<DiscreteMask> {
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

    pub fn greedy_action(&mut self, rng: &mut StdRng, obs: &Tensor, mask: DiscreteMask) -> Result<u32> {
        let greedy_actions = self.greedy_actions(obs, mask)?;
        let n = greedy_actions.n_possible_actions();
        let r = rng.random_range(0..n);
        Ok(greedy_actions.iter().nth(r).unwrap())
    }
}

impl<M: Module> Module for DiscreteQNet<M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.network.forward(xs)
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