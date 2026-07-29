use std::collections::VecDeque;

use candle_core::{Device, Result};
use rand::{SeedableRng, rngs::StdRng, seq::index::sample};

use crate::{components::transition::{BatchedTransition, Transition}, traits::Batchable};

pub struct ReplayBuffer<Obs, Action, Mask = (), Extra = ()>
where 
    Obs: Batchable + Clone,
    Mask: Batchable + Clone,
    Action: Batchable + Clone,
    Extra: Batchable + Clone,
{
    capacity: usize,
    rng: StdRng,

    observations: VecDeque<Obs>,
    masks: VecDeque<Mask>,
    actions: VecDeque<Action>,
    next_observations: VecDeque<Obs>,
    next_masks: VecDeque<Mask>,
    rewards: VecDeque<f32>,
    terminates: VecDeque<bool>,
    truncates: VecDeque<bool>,
    extras: VecDeque<Extra>,
}

impl<Obs, Action, Mask, Extra> ReplayBuffer<Obs, Action, Mask, Extra>
where
    Obs: Batchable + Clone,
    Mask: Batchable + Clone,
    Action: Batchable + Clone,
    Extra: Batchable + Clone,
{
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            capacity,
            rng: StdRng::seed_from_u64(seed),

            observations: VecDeque::with_capacity(capacity),
            masks: VecDeque::with_capacity(capacity),
            actions: VecDeque::with_capacity(capacity),
            next_observations: VecDeque::with_capacity(capacity),
            next_masks: VecDeque::with_capacity(capacity),
            rewards: VecDeque::with_capacity(capacity),
            terminates: VecDeque::with_capacity(capacity),
            truncates: VecDeque::with_capacity(capacity),
            extras: VecDeque::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, transition: Transition<Obs, Action, Mask, Extra>) {
        if self.capacity <= self.len() {
            // pop front
            self.observations.pop_front();
            self.next_observations.pop_front();
            self.actions.pop_front();
            self.rewards.pop_front();
            self.terminates.pop_front();
            self.truncates.pop_front();
            self.extras.pop_front();
            self.masks.pop_front();
            self.next_masks.pop_front();
        }

        let (obs, mask) = transition.masked_observation;
        let action = transition.action;
        let reward = transition.reward;
        let (next_obs, next_mask) = transition.next_masked_observation;
        let terminate = transition.terminate;
        let truncate = transition.truncate;
        let extra = transition.extra;

        self.observations.push_back(obs);
        self.actions.push_back(action);
        self.rewards.push_back(reward);
        self.terminates.push_back(terminate);
        self.truncates.push_back(truncate);
        self.extras.push_back(extra);
        self.next_observations.push_back(next_obs);
        self.masks.push_back(mask);
        self.next_masks.push_back(next_mask);
        
    }

    pub fn sample(&mut self, batch_size: usize, device: &Device) -> Result<Option<BatchedTransition<Obs::Batched, Action::Batched, Mask::Batched, Extra::Batched>>> {
        let len = self.len();
        if len < batch_size {
            return Ok(None);
        }

        let indices = sample(&mut self.rng, len, batch_size);
        let (observations, masks, actions, rewards, next_observations, next_masks, terminates, truncates, extras)
        : (Vec<Obs>, Vec<Mask>, Vec<Action>, Vec<f32>, Vec<Obs>, Vec<Mask>, Vec<f32>, Vec<f32>, Vec<Extra>)
        = indices.iter().map(|i| {
            (
                self.observations[i].clone(),
                self.masks[i].clone(),
                self.actions[i].clone(),
                self.rewards[i].clone(),
                self.next_observations[i].clone(),
                self.next_masks[i].clone(),
                if self.terminates[i] { 1.0 } else { 0.0 },
                if self.truncates[i] { 1.0 } else { 0.0 },
                self.extras[i].clone()
            )
        }).collect();
        
        Ok(BatchedTransition {
            observations: Obs::batch(observations, device)?,
            masks: Mask::batch(masks, device)?,
            actions: Action::batch(actions, device)?,
            rewards: f32::batch(rewards, device)?,
            next_observations: Obs::batch(next_observations, device)?,
            next_masks: Mask::batch(next_masks, device)?,
            terminates: f32::batch(terminates, device)?,
            truncates: f32::batch(truncates, device)?,
            extras: Extra::batch(extras, device)?,
        }.into())
        
    }

    pub fn len(&self) -> usize { self.observations.len() }

    // for debug
    pub fn observations(&self) -> &VecDeque<Obs> { &self.observations }
    // for debug
    pub fn actions(&self) -> &VecDeque<Action> { &self.actions }
}