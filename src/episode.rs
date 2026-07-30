use candle_core::Result;

use crate::{agent::Agent, components::{mask::discretemask::DiscreteMask, transition::Transition}, environment::Environment, traits::Batchable};
// test structure

pub trait EpisodeIterator<E: Environment>{
    type Extra: Batchable + Clone;

    fn new<A>(env: &mut E, agent: &mut A, max_steps: Option<usize>) -> Self
    where
        A: Agent<E::Obs, Action = E::Action, Mask = E::Mask, Extra = Self::Extra>;

    fn reset<A>(&mut self, env: &mut E, agent: &mut A)
    where
        A: Agent<E::Obs, Action = E::Action, Mask = E::Mask, Extra = Self::Extra>;

    fn step<A>(&mut self, env: &mut E, agent: &mut A) -> Result<Option<Transition<E::Obs, E::Action, E::Mask, A::Extra>>>
    where
        A: Agent<E::Obs, Action = E::Action, Mask = E::Mask, Extra = Self::Extra>;

    fn steps(&self) -> usize;
}

pub struct BasicEpisodeIterator<E: Environment> {
    obs: E::Obs,
    mask: E::Mask,
    max_steps: Option<usize>,
    steps: usize,
    done: bool,
}

impl<E: Environment> EpisodeIterator<E> for BasicEpisodeIterator<E> {
    type Extra = ();

    fn new<A>(env: &mut E, agent: &mut A, max_steps: Option<usize>) -> Self
    where
        A: Agent<E::Obs, Action = E::Action, Mask = E::Mask, Extra = Self::Extra>
    {
        let (obs, mask) = env.reset().unwrap();

        Self {
            obs,
            mask,
            max_steps,
            steps: 0,
            done: false,
        }
    }

    fn reset<A>(&mut self, env: &mut E, agent: &mut A)
    where
        A: Agent<E::Obs, Action = E::Action, Mask = E::Mask, Extra = Self::Extra>
    {
        let (obs, mask) = env.reset().unwrap();
        self.obs = obs;
        self.mask = mask;
        self.steps = 0;
        self.done = false;
    }

    fn step<A>(&mut self, env: &mut E, agent: &mut A) -> Result<Option<Transition<E::Obs, E::Action, E::Mask, A::Extra>>>
    where
        A: Agent<E::Obs, Action = E::Action, Mask = E::Mask, Extra = Self::Extra>
    {
        if self.done { return Ok(None) }

        let action = agent.explore(&self.obs, self.mask.clone()).unwrap();
        let (next_obs, next_mask, reward, terminate, truncate) = env.step(action.clone()).unwrap();

        let mut transition = Transition {
            masked_observation: (self.obs.clone(), self.mask.clone()),
            action,
            reward,
            next_masked_observation: (next_obs.clone(), next_mask.clone()),
            terminate,
            truncate,
            extra: (),
        };
        self.steps += 1;
        self.obs = next_obs;
        self.mask = next_mask;

        if let Some(max_steps) = self.max_steps {
            if max_steps <= self.steps { transition.truncate = true }
        }

        if transition.terminate || transition.truncate { self.done = true }

        Ok(Some(transition))
    }

    fn steps(&self) -> usize { self.steps }
}
