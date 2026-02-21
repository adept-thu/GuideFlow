      
import flax.linen as nn
import jax
import jax.numpy as jnp

# Import your models here
from models import models_dit


def generate(variable, model, rng, n_sample, config, class_idx=None):
  """
  Generate samples from the model
  ---
  return shape: (n_sample, H, W, C)
  """
  num_steps = config.sampling.num_steps
  num_classes = config.dataset.num_classes
  img_size, img_channels = config.dataset.image_size, config.dataset.image_channels

  # prepare step schedule
  t_steps = model.apply(
      {},
      method=model.sampling_schedule(),
  )

  # initialize noise
  x_shape = (n_sample, img_size, img_size, img_channels)
  rng_xt, rng = jax.random.split(rng, 2)

  z_t = jax.random.normal(rng_xt, x_shape, dtype=model.dtype)   # z_t ~ N(0, 1), sample initial noise

  rng, rng_sample = jax.random.split(rng, 2)
  if class_idx is None:
    labels = jax.random.randint(rng_sample, (n_sample,), 0, num_classes)
  else:
    labels = jnp.ones((n_sample,), dtype=jnp.int32) * class_idx

  def step_fn(i, inputs):
    x_i, rng = inputs
    rng_step = jax.random.fold_in(rng, i)
    rng_z, _ = jax.random.split(rng_step, 2)

    x_i = model.apply(
      variable, x_i, labels, i, t_steps,
      method=model.sample_one_step,
      rngs=dict(gen=rng_z,),
    )
    
    outputs = (x_i, rng)
    return outputs

  outputs = jax.lax.fori_loop(0, num_steps, step_fn, (z_t, rng))
  images = outputs[0]
  return images


class MeanFlow(nn.Module):
  """MeanFlow"""

  # Model and dataset
  model_str:              str
  model_config:           dict
  dtype =                 jnp.float32

  num_classes:            int = 1000

  # Noise distribution
  noise_dist:             str   = 'logit_normal'
  P_mean:                 float = -0.4
  P_std:                  float = 1.0

  # Loss
  data_proportion:        float = 0.75

  guidance_eq:            str   = 'cfg'
  omega:                  float = 1.0
  kappa:                  float = 0.5
  class_dropout_prob:     float = 0.1

  t_start:                float = 0.0
  t_end:                  float = 1.0

  # Training dynamics
  norm_p:                 float = 1.0
  norm_eps:               float = 0.01

  # Inference setups
  seed:                   int = 0
  num_steps:              int = 1
  schedule:               str = 'default'
  sampling_timesteps:     jnp.ndarray = None

  def setup(self):
    model_str = self.model_str

    net_fn = getattr(models_dit, model_str)
    self.net = net_fn(name="net", class_dropout_prob=0.0)

  #######################################################
  # Solver
  #######################################################

  def sample_one_step(self, z_t, labels, i, t_steps):
    t = t_steps[i]
    r = t_steps[i + 1]

    t = jnp.repeat(t, z_t.shape[0])
    r = jnp.repeat(r, z_t.shape[0])
    
    return self.solver_step(z_t, t, r, labels)

  def solver_step(self, z_t, t, r, labels):
    u = self.u_fn(z_t, t=t, h=(t - r), y=labels, train=False)
    return z_t - jnp.einsum('n,n...->n...', t - r, u)
  
  def sampling_schedule(self):
    if self.schedule == 'default':
      return self._default_schedule
    else:
      raise ValueError(f"Unknown schedule: {self.schedule}")

  def _default_schedule(self):
    if self.sampling_timesteps is None:
      return jnp.array([1.0, 0.0])
    return self.sampling_timesteps

  #######################################################
  # Schedule
  #######################################################

  def noise_distribution(self):
    if self.noise_dist == 'logit_normal':
        return self._logit_normal_dist
    elif self.noise_dist == 'uniform':
        return self._uniform_dist
    else:
        raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

  def _logit_normal_dist(self, bz):
    rnd_normal = jax.random.normal(self.make_rng('gen'), [bz, 1, 1, 1], dtype=self.dtype)
    return nn.sigmoid(
      rnd_normal * self.P_std + self.P_mean
      )
  
  def _uniform_dist(self, bz):
    return jax.random.uniform(self.make_rng('gen'), [bz, 1, 1, 1], dtype=self.dtype)
  
  def sample_tr(self, bz):
    t = self.noise_distribution()(bz)
    r = self.noise_distribution()(bz)
    t, r = jnp.maximum(t, r), jnp.minimum(t, r)

    data_size = int(bz * self.data_proportion)
    zero_mask = jnp.arange(bz) < data_size
    zero_mask = zero_mask.reshape(bz, 1, 1, 1)
    r = jnp.where(zero_mask, t, r)

    return t, r

  #######################################################
  # Training Utils & Guidance
  #######################################################

  def u_fn(self, x, t, h, y, train=True):
    bz = x.shape[0]
    return self.net(x, t.reshape(bz), h.reshape(bz), y, train=train, key=self.make_rng('gen'))
  
  def v_fn(self, x, t, y, train=False):
    h = jnp.zeros_like(t)
    return self.u_fn(x, t, h, y=y, train=train)
  
  def guidance_fn(self, v_t, z_t, t, y, train=False):
    if self.guidance_eq == 'cfg' and self.kappa == 0:
      y_null = jnp.array([self.num_classes] * z_t.shape[0]) # for unconditional velocity
      v_uncond = self.v_fn(z_t, t, y=y_null, train=train)

      omega = jnp.where((t >= self.t_start) & (t <= self.t_end), self.omega, 1.0)
      v_g   = v_uncond + omega * (v_t - v_uncond)
    elif self.guidance_eq == 'cfg' and self.kappa > 0:
      y_null   = jnp.array([self.num_classes] * z_t.shape[0])
      v_uncond = self.v_fn(z_t, t, y=y_null, train=train)
      v_cond   = self.v_fn(z_t, t, y=y, train=train)

      omega = jnp.where((t >= self.t_start) & (t <= self.t_end), self.omega, 1.0)
      kappa = jnp.where((t >= self.t_start) & (t <= self.t_end), self.kappa, 0.0)
      v_g   = omega * v_t + (1 - omega - kappa) * v_uncond + kappa * v_cond
    else:
      v_g   = v_t
    
    return v_g

  def cond_drop(self, v_t, v_g, labels):
    bz = v_t.shape[0]

    rand_mask = jax.random.uniform(self.make_rng('gen'), shape=(bz,)) < self.class_dropout_prob
    num_drop  = jnp.sum(rand_mask).astype(jnp.int32)
    drop_mask = jnp.arange(bz)[:, None, None, None] < num_drop

    y_inp = jnp.where(drop_mask.reshape(bz,), self.num_classes, labels)
    v_g   = jnp.where(drop_mask, v_t, v_g)
    return y_inp, v_g
  
  #######################################################
  # Forward Pass and Loss
  #######################################################

  def forward(self, imgs, labels, train=True):
    x  = imgs.astype(self.dtype)
    bz = imgs.shape[0]

    # -----------------------------------------------------------------
    # Instantaneous velocity
    t, r = self.sample_tr(bz)

    e   = jax.random.normal(self.make_rng('gen'), x.shape, dtype=self.dtype)
    z_t = (1 - t) * x + t * e
    v   = e - x

    # Guided velocity
    v_g = self.guidance_fn(v, z_t, t, labels, train=False) if self.guidance_eq else v

    # Cond dropout (dropout class labels)
    y_inp, v_g = self.cond_drop(v, v_g, labels)

    # -----------------------------------------------------------------
    # Compute u_tr (average velocity) and du_dt using jvp
    def u_fn(z_t, t, r):
      return self.u_fn(z_t, t, t - r, y=y_inp, train=train)

    dt_dt = jnp.ones_like(t)
    dr_dt = jnp.zeros_like(t)
    u, du_dt = jax.jvp(u_fn, (z_t, t, r), (v_g, dt_dt, dr_dt))

    # -----------------------------------------------------------------
    # Compute loss
    u_tgt = v_g - jnp.clip(t - r, a_min=0.0, a_max=1.0) * du_dt
    u_tgt = jax.lax.stop_gradient(u_tgt)

    loss = (u - u_tgt) ** 2
    loss = jnp.sum(loss, axis=(1, 2, 3)) # sum over pixels

    # Adaptive weighting
    adp_wt = (loss + self.norm_eps) ** self.norm_p
    loss = loss / jax.lax.stop_gradient(adp_wt)

    # -----------------------------------------------------------------
    loss = loss.mean()  # mean over batch

    # Velocity loss, monitoring only
    v_loss = (u - v) ** 2
    v_loss = jnp.sum(v_loss, axis=(1, 2, 3))

    dict_losses = {
      'loss': loss,
      'v_loss': v_loss,
    }
    return loss, dict_losses

  def __call__(self, x, t, y, train=False, key=None):
    return self.net(x, t, t, y, key=key, train=train) # initialization only