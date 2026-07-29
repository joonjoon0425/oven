use std::ops::Not;

use candle_core::{Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct DiscreteMask {
    pub mask: u64,
    pub n_actions: usize,   
}

pub struct BitSetIterator {
    mask: u64,
}

impl DiscreteMask {
    pub fn all_enabled(n_actions: usize) -> Self {
        let mask = if n_actions >= 128 { u64::MAX } else { (1u64 << n_actions) - 1 };
        DiscreteMask { mask, n_actions }
    }

    pub fn all_disabled(n_actions: usize) -> Self {
        DiscreteMask { mask: 0, n_actions, }
    }

    pub fn enable(&mut self, index: usize) {
        self.mask |= 1u64 << index;
    }

    pub fn disable(&mut self, index: usize) {
        self.mask &= (1u64 << index).not();
    }

    pub fn is_enabled(&self, index: usize) -> bool {
        (self.mask >> index) & 1u64 == 1u64
    }

    pub fn disable_all(&mut self) {
        self.mask = 0;
    }

    pub fn iter(&self) -> BitSetIterator {
        BitSetIterator { mask: self.mask }
    }
}

impl BitSetIterator {
    pub fn new(n: usize) -> BitSetIterator {
        let mask = if n >= 128 { u64::MAX } else { (1u64 << n) - 1 };
        BitSetIterator { mask }
    }
}

impl Iterator for BitSetIterator {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.mask == 0 {
            return None;
        } else {
            let ret = self.mask.trailing_zeros();
            self.mask &= self.mask - 1;
            return Some(ret);
        }
    }
}

impl ExactSizeIterator for BitSetIterator {
    fn len(&self) -> usize {
        self.mask.count_ones() as usize
    }
}

impl DiscreteMask {
    pub fn to_tensor(self, device: &Device) -> Result<Tensor> {
        let n = self.n_actions;
        let mut data = vec![0f32; n];
        for action in self.iter() {
            data[action as usize] = 1f32;
        }
        Tensor::from_vec(data, (n, ), device)
    }

    pub fn n_possible_actions(&self) -> usize {
        self.mask.count_ones() as usize
    }
}