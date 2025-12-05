# CUDA_VISIBLE_DEVICES=0 python -m baselines.dte_ppo_rnn --config_file dte_ppo_rnn.yaml
"""
Code is adapted from the IPPO RNN implementation of JaxMARL (https://github.com/FLAIROx/JaxMARL/tree/main) 
Credit goes to the original authors: Rutherford et al.
"""

# ===========================
# Imports and Configuration
# ===========================
import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import functools
import yaml
from typing import Sequence, NamedTuple, Dict

import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import optax
import distrax

import wandb

from jaxmarl.wrappers.baselines import LogWrapper
from craftax.craftax_env import make_craftax_env_from_name

# ===========================
# Model Definitions
# ===========================
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

# ===========================
# Data Structures and Utilities
# ===========================
class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_agents, num_envs):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_agents, num_envs, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

# ===========================
# Training Function
# ===========================
def make_train(config, env):
    config["LR"] = float(config["LR"])
    config["TOTAL_TIMESTEPS"] = int(config["TOTAL_TIMESTEPS"])
    
    # --- MODIFIED: Num_Actors is just Num_Envs now, as we VMAP over agents ---
    config["NUM_ACTORS"] = config["NUM_ENVS"] 
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # Minibatch size is per agent
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)

        # Shape: (Num_Agents, 1, Num_Envs, Obs)
        init_x = (
            jnp.zeros(
                (env.num_agents, 1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((env.num_agents, 1, config["NUM_ENVS"])),
        )
        
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        # Duplicate hstate for number of agents: [Num_Agents, Num_Envs, D_Model]
        init_hstate = jnp.tile(init_hstate[None, ...], (env.num_agents, 1, 1))

        # VMAP Network Init over Agents (Axis 0) ---
        # We split RNG keys so each agent gets different random initialization
        init_rngs = jax.random.split(_rng, env.num_agents)
        network_params = jax.vmap(network.init, in_axes=(0, 0, 0))(init_rngs, init_hstate, init_x)
        
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = jax.vmap(
                    lambda p: TrainState.create(
                        apply_fn=network.apply,
                        params=p,
                        tx=tx,
                    )
                )(network_params)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # last_done = jnp.tile(last_done[None, ...], (env.num_agents, 1))
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, env.num_agents, config["NUM_ENVS"])

                # Add fake Time Dimension [:, None, ...] ---
                # obs_batch and hstate: [Agents, Envs, Obs] -> [Agents, 1, Envs, Obs]
                ac_in = (obs_batch[:, None, ...], last_done[:, None, ...])

                # print(f"DEBUG SHAPE - hstate: {hstate.shape}")
                # print(f"DEBUG SHAPE - ac_in:  {ac_in[0].shape}")
                # print(f"DEBUG SHAPE - last_done:  {ac_in[1].shape}")
                # Vmap Apply: Map over Axis 0 of params, hstate, and ac_in
                hstate, pi, value = jax.vmap(network.apply, in_axes=(0, 0, 0))(
                    train_state.params, hstate, ac_in
                )
                # Sampling: Need distinct keys for each agent to avoid correlated noise
                rng_step = jax.random.split(_rng, env.num_agents)
                action = jax.vmap(lambda p, r: p.sample(seed=r))(pi, rng_step)
                log_prob = pi.log_prob(action)

                # Env step expects {agent: [Batch]}
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()} # No squeeze needed if dimensionality matches

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                done_batch = batchify(done, env.agents, env.num_agents, config["NUM_ENVS"]).squeeze()
                global_done = jnp.tile(done["__all__"][None, :], (env.num_agents, 1))
                transition = Transition(
                    global_done,
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, env.num_agents, config["NUM_ENVS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, env.num_agents, config["NUM_ENVS"])
            
            ac_in = (last_obs_batch[:, None, ...], last_done[:, None, ...])

            _, _, last_val = jax.vmap(network.apply, in_axes=(0, 0, 0))(
                train_state.params, hstate, ac_in
            )
            last_val = last_val.squeeze()
            # print(f"DEBUG - last_val {last_val.shape}")

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    # print(f"DEBUG - reward {reward.shape}")
                    # print(f"DEBUG - next_value {next_value.shape}")
                    # print(f"DEBUG - value {value.shape}")
                    # print(f"DEBUG - done {done.shape}")
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # Data comes in as (Batch, Time, ...). 
                        # Scan expects Time to be Axis 0.
                        
                        # 1. Transpose Inputs: (Batch, Time) -> (Time, Batch)
                        obs_in = traj_batch.obs.swapaxes(0, 1)   # (Time, Batch, Obs)
                        dones_in = traj_batch.done.swapaxes(0, 1) # (Time, Batch)

                        # RERUN NETWORK
                        # init_hstate: (Batch, D_Model)
                        # Input tuple: ((Time, Batch, Obs), (Time, Batch))
                        _, pi, value = network.apply(
                            params,
                            init_hstate, 
                            (obs_in, dones_in),
                        )
                        
                        # Network output is (Time, Batch, ...). 
                        # We swap back to (Batch, Time, ...) to match targets.
                        value = value.swapaxes(0, 1) 
                        
                        # Log Prob Calculation
                        # Action: (Batch, Time) -> (Time, Batch) for log_prob computation
                        action_in = traj_batch.action.swapaxes(0, 1)
                        log_prob = pi.log_prob(action_in).swapaxes(0, 1) # Swap result back
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, 0.0, 0.0)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # 1. Strip 'info' from traj_batch for the update loop.
                # 'info' lacks the Agent dimension and causes vmap crashes.
                # It is not needed for the loss calculation.
                traj_batch_clean = traj_batch._replace(info={})

                # From (Time, Agents, Envs, ...) -> (Agents, Envs, Time, ...)
                traj_batch_t = jax.tree.map(
                    lambda x: x.swapaxes(0, 1).swapaxes(1, 2), 
                    traj_batch_clean 
                )

                # (Time, Agents, Envs) -> (Agents, Envs, Time)
                adv_t = advantages.swapaxes(0, 1).swapaxes(1, 2)
                tar_t = targets.swapaxes(0, 1).swapaxes(1, 2)

                batch = (
                    init_hstate,
                    traj_batch_t,
                    adv_t,
                    tar_t,
                )
                # 5. Shuffle Environments (Axis 1)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )   
                # We want the final shape: (Num_Minibatches, Agents, Minibatch_Size, Time, ...)
                def reshape_minibatch(x):
                    # Input x: (Agents, Envs, [Time], ...)
                    # We split Axis 1 (Envs) into (Num_MB, MB_Size)
                    new_shape = list(x.shape)
                    new_shape[1] = config["NUM_MINIBATCHES"]
                    new_shape.insert(2, -1) # Calculate MB_Size automatically
                    
                    # Reshape: (Agents, Num_MB, MB_Size, [Time], ...)
                    reshaped = x.reshape(new_shape)
                    
                    # Swap axes to put Num_MB first: (Num_MB, Agents, MB_Size, ...)
                    return reshaped.swapaxes(0, 1)

                minibatches = jax.tree.map(reshape_minibatch, shuffled_batch)

                # --- MODIFIED: VMAP the update function over agents ---
                # We scan over minibatches, and VMAP over agents inside
                def _scan_update(train_state, minibatch):
                    # minibatch is tuple of [Num_Agents, MB_Size, ...]
                    # We map over axis 0
                    train_state, total_loss = jax.vmap(_update_minbatch, in_axes=(0, 0))(train_state, minibatch)
                    return train_state, total_loss

                train_state, total_loss = jax.lax.scan(
                    _scan_update, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info

            ratio_0 = loss_info[1][3].at[0,0].get().mean()

            # 1. Average over Time (Epochs & Minibatches), keeping Agent dimension
            # Input shape: [Epochs, Minibatches, Agents]
            # Output shape: [Agents]
            loss_info_per_agent = jax.tree.map(lambda x: x.mean(axis=(0, 1)), loss_info)
            
            # 2. Also calculate the Global Mean (for general overview)
            loss_info_global = jax.tree.map(lambda x: x.mean(), loss_info)

            metric["update_steps"] = update_steps
            
            # 3. Create the Base Global Metrics
            metric["loss"] = {
                "total_loss": loss_info_global[0],
                "value_loss": loss_info_global[1][0],
                "actor_loss": loss_info_global[1][1],
                "entropy": loss_info_global[1][2],
                "ratio": loss_info_global[1][3],
                "approx_kl": loss_info_global[1][4],
                "clip_frac": loss_info_global[1][5],
            }

            # 4. Add Per-Agent Metrics dynamically
            # This creates keys like: "agent_0/total_loss", "agent_0/entropy", etc.
            num_agents = env.num_agents # or config["NUM_AGENTS"]
            for i in range(num_agents):
                agent_metrics = {
                    f"agent_{i}/total_loss": loss_info_per_agent[0][i],
                    f"agent_{i}/value_loss": loss_info_per_agent[1][0][i],
                    f"agent_{i}/actor_loss": loss_info_per_agent[1][1][i],
                    f"agent_{i}/entropy": loss_info_per_agent[1][2][i],
                    f"agent_{i}/approx_kl": loss_info_per_agent[1][4][i],
                }
                metric["loss"].update(agent_metrics)
                
            rng = update_state[-1]

            def callback(metrics, actor_state: TrainState, step):
                env_step = (
                    metrics["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                )
                to_log = {
                    "env_step": env_step,
                    **metrics["loss"],
                }
                if metrics["returned_episode"].any():
                    to_log.update(jax.tree.map(
                        lambda x: x[metrics["returned_episode"]].mean(),
                        metrics["user_info"]
                    ))
                    to_log["episode_lengths"] = metrics["returned_episode_lengths"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                    to_log["episode_returns"] = metrics["returned_episode_returns"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                # print(to_log)
                wandb.log(to_log, step=metrics["update_steps"])

            jax.experimental.io_callback(callback, None, metric, train_state, update_steps)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

# ===========================
# Main Run Function
# ===========================
def single_run(config):
    alg_name = config.get("ALG_NAME", "ippo-rnn")
    env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    env = make_craftax_env_from_name(env_name)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config["RUN_NAME"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    # =========================================================================
    # INSERT THIS BLOCK TO COUNT PARAMETERS
    # =========================================================================
    print("Initializing model to count parameters...")
    # 1. Create a dummy network instance
    dummy_network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    
    # 2. Create dummy inputs (Batch size 1 is enough for checking)
    dummy_rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape[0])) # [1, Batch, Obs]
    print(f"OBS SIZE {dummy_obs.shape}")
    dummy_dones = jnp.zeros((1, 1))                                              # [1, Batch]
    dummy_memory = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])             # [Batch, D_Model]
    
    # 3. Initialize parameters
    dummy_params = dummy_network.init(dummy_rng, dummy_memory, (dummy_obs, dummy_dones))
    
    # 4. Count and Print
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(dummy_params))
    print(f"\n{'='*40}")
    print(f" TOTAL TRAINABLE PARAMETERS: {total_params:,}")
    print(f"{'='*40}\n")
    # =========================================================================

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Name of the config YAML file (in baselines/config/)")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "config", args.config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    single_run(config)


if __name__ == "__main__":
    main()
