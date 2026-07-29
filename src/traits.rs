use candle_core::{Device, Result, Tensor};

pub trait Batchable: Sized {
    type Batched;
    
    fn batch(storage: &[Self], device: &Device) -> Result<Self::Batched>;
}

impl Batchable for Tensor {
    type Batched = Tensor;
    
    fn batch(storage: &[Self], device: &Device) -> Result<Self::Batched> {
        Tensor::stack(storage, 0)?.to_device(device)
    }
}

impl Batchable for () {
    type Batched = ();

    fn batch(_: &[Self], _: &Device) -> Result<Self::Batched> {
        Ok(())
    }
}

impl Batchable for f32 {
    type Batched = Tensor;

    fn batch(storage: &[Self], device: &Device) -> Result<Self::Batched> {
        // the cloning occurs here... I guess
        Tensor::from_slice(storage, (storage.len(), ), device)
    }
}

impl Batchable for u32 {
    type Batched = Tensor;

    fn batch(storage: &[Self], device: &Device) -> Result<Self::Batched> {
        Tensor::from_slice(storage, (storage.len(), ), device)
    }
}