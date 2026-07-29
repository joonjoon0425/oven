use candle_core::{Device, Result, Tensor};

use crate::components::mask::discretemask::DiscreteMask;

pub trait Batchable: Sized {
    type Batched;
    
    fn batch(storage: Vec<Self>, device: &Device) -> Result<Self::Batched>;
}

impl Batchable for Tensor {
    type Batched = Tensor;
    
    fn batch(storage: Vec<Self>, device: &Device) -> Result<Self::Batched> {
        Tensor::stack(&storage, 0)?.to_device(device)
    }
}

impl Batchable for () {
    type Batched = ();

    fn batch(_: Vec<Self>, _: &Device) -> Result<Self::Batched> {
        Ok(())
    }
}

impl Batchable for f32 {
    type Batched = Tensor;

    fn batch(storage: Vec<Self>, device: &Device) -> Result<Self::Batched> {
        // the cloning occurs here... I guess
        let len = storage.len();
        Tensor::from_vec(storage, (len, ), device)
    }
}

impl Batchable for u32 {
    type Batched = Tensor;

    fn batch(storage: Vec<Self>, device: &Device) -> Result<Self::Batched> {
        let len = storage.len();
        Tensor::from_vec(storage, (len, ), device)
    }
}

impl Batchable for DiscreteMask {
    type Batched = Tensor;

    fn batch(storage: Vec<Self>, device: &Device) -> Result<Self::Batched> {
        let n = storage[0].n_actions;
        let mut data = vec![0f32; storage.len() * n];
        for (i, mask) in storage.iter().enumerate() {
            for action in mask.iter() {
                data[i * n + action as usize] = 1f32;
            }
        }
        Tensor::from_vec(data, (storage.len(), n), device)
    }
}