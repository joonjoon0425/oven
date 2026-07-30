use candle_core::{DType, Device, Result, Tensor};

/// 클래식 CartPole-v1과 동일한 물리 파라미터
pub struct CartPoleEnv {
    device: Device,

    // 물리 상수 (OpenAI Gym CartPole-v1 기준)
    gravity: f64,
    masscart: f64,
    masspole: f64,
    total_mass: f64,
    length: f64,        // 실제로는 pole 길이의 절반
    polemass_length: f64,
    force_mag: f64,
    tau: f64,           // 시뮬레이션 timestep (초)

    // 종료 조건
    theta_threshold_radians: f64,
    x_threshold: f64,
    max_steps: usize,

    // 내부 상태: [x, x_dot, theta, theta_dot]
    state: [f64; 4],
    steps: usize,

    rng: StdRng,
}

use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

use crate::{components::mask::DiscreteMask, environment::Environment};

impl CartPoleEnv {
    pub fn new(device: &Device, seed: u64) -> Self {
        Self {
            device: device.clone(),
            gravity: 9.8,
            masscart: 1.0,
            masspole: 0.1,
            total_mass: 1.0 + 0.1,
            length: 0.5,
            polemass_length: 0.1 * 0.5,
            force_mag: 10.0,
            tau: 0.02,
            theta_threshold_radians: 12.0 * 2.0 * std::f64::consts::PI / 360.0, // ±12도
            x_threshold: 2.4,
            max_steps: 500,
            state: [0.0; 4],
            steps: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn state_to_tensor(&self) -> Result<Tensor> {
        Tensor::from_slice(&self.state, (4,), &self.device)?.to_dtype(DType::F32)
    }

    pub fn action_space_size(&self) -> usize {
        2
    }

    pub fn obs_dim(&self) -> usize {
        4
    }
}

impl Environment for CartPoleEnv {
    type Obs = Tensor;
    type Action = u32;
    type Mask = DiscreteMask;

    /// 환경을 초기화하고 초기 관측값을 (1, 4) 텐서로 반환
    fn reset(&mut self) -> Result<(Tensor, Self::Mask)> {
        // Gym과 동일하게 [-0.05, 0.05] uniform 초기화
        for s in self.state.iter_mut() {
            *s = self.rng.random_range(-0.05..0.05);
        }
        self.steps = 0;
        Ok((self.state_to_tensor()?, DiscreteMask::all_enabled(self.action_space_size())))
    }

    /// action: 0 = 왼쪽으로 힘, 1 = 오른쪽으로 힘
    /// 반환: (obs, reward, terminated, truncated)
    fn step(&mut self, action: u32) -> Result<(Tensor, Self::Mask, f32, bool, bool)> {
        assert!(action == 0 || action == 1, "action은 0 또는 1이어야 함");

        let [x, x_dot, theta, theta_dot] = self.state;

        let force = if action == 1 { self.force_mag } else { -self.force_mag };
        let costheta = theta.cos();
        let sintheta = theta.sin();

        // CartPole 운동방정식 (Gym 구현과 동일)
        let temp = (force + self.polemass_length * theta_dot.powi(2) * sintheta) / self.total_mass;
        let theta_acc = (self.gravity * sintheta - costheta * temp)
            / (self.length * (4.0 / 3.0 - self.masspole * costheta.powi(2) / self.total_mass));
        let x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass;

        // Euler integration
        let new_x = x + self.tau * x_dot;
        let new_x_dot = x_dot + self.tau * x_acc;
        let new_theta = theta + self.tau * theta_dot;
        let new_theta_dot = theta_dot + self.tau * theta_acc;

        self.state = [new_x, new_x_dot, new_theta, new_theta_dot];
        self.steps += 1;

        let terminated = new_x < -self.x_threshold
            || new_x > self.x_threshold
            || new_theta < -self.theta_threshold_radians
            || new_theta > self.theta_threshold_radians;

        let truncated = !terminated && self.steps >= self.max_steps;

        // Gym 관례: 살아있으면 매 스텝 +1 reward
        let reward = 1.0f32;

        Ok((self.state_to_tensor()?, DiscreteMask::all_enabled(self.action_space_size()), reward, terminated, truncated))
    }
}