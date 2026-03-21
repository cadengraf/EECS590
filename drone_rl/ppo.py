import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions

# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

class DeliveryEnv:
    """
    Thin gym-style wrapper around the bw_map grid.
    State: (row, col, has_package) flattened to an integer observation.
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E

    def __init__(self, bw_map, start_pos, package_pos, delivery_pos):
        self.bw_map = bw_map
        self.start_pos = start_pos
        self.package_pos = package_pos
        self.delivery_pos = delivery_pos
        self.rows, self.cols = bw_map.shape
        self.lane_set = set(map(tuple, np.argwhere(bw_map == 1)))
        self.reset()

    # --- Observation helpers ------------------------------------------------

    def _encode(self, r, c, has_pkg):
        """Flat integer obs: uniquely indexes every (r, c, has_pkg) triple."""
        return (int(has_pkg) * self.rows * self.cols) + r * self.cols + c

    @property
    def obs_dim(self):
        return 2 * self.rows * self.cols  # two 'layers': no-pkg / has-pkg

    @property
    def n_actions(self):
        return len(self.ACTIONS)

    # --- Core env API -------------------------------------------------------

    def reset(self):
        r, c = self.start_pos
        self.state = (r, c, False)
        return self._encode(r, c, False)

    def step(self, action):
        r, c, has_pkg = self.state
        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc

        # Walls: stay in place
        if (nr, nc) not in self.lane_set:
            nr, nc = r, c

        reward = -0.01  # small time penalty each step

        if not has_pkg:
            if (nr, nc) == self.package_pos:
                has_pkg = True
                reward += 1.0   # pickup bonus
        else:
            if (nr, nc) == self.delivery_pos:
                self.state = (nr, nc, has_pkg)
                return self._encode(nr, nc, has_pkg), reward + 10.0, True

        self.state = (nr, nc, has_pkg)
        return self._encode(nr, nc, has_pkg), reward, False

    def full_state(self):
        return self.state


# ---------------------------------------------------------------------------
# ACTOR-CRITIC NETWORK
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared MLP backbone with two heads:
      - actor:  outputs action logits  → policy π(a|s)
      - critic: outputs scalar value   → V(s)
    """

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def _one_hot(self, obs_idx, obs_dim):
        x = torch.zeros(obs_dim)
        x[obs_idx] = 1.0
        return x

    def forward(self, x):
        h = self.backbone(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs_idx, obs_dim):
        """Sample an action and return (action, log_prob, value)."""
        x = self._one_hot(obs_idx, obs_dim).unsqueeze(0)
        logits, value = self(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs_batch, act_batch):
        """Evaluate a batch of (obs, action) pairs for the PPO update."""
        logits, values = self(obs_batch)
        dist     = Categorical(logits=logits)
        log_probs = dist.log_prob(act_batch)
        entropy   = dist.entropy()
        return log_probs, values, entropy


# ---------------------------------------------------------------------------
# PPO AGENT
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Proximal Policy Optimization with:
      - Clipped surrogate objective
      - Value function loss (MSE)
      - Entropy bonus
      - GAE advantage estimation
    """

    def __init__(
        self,
        obs_dim,
        n_actions,
        lr           = 3e-4,
        gamma        = 0.99,
        lam          = 0.95,   # GAE lambda
        clip_eps     = 0.2,
        vf_coef      = 0.5,
        ent_coef     = 0.01,
        n_epochs     = 4,
        batch_size   = 64,
    ):
        self.obs_dim    = obs_dim
        self.gamma      = gamma
        self.lam        = lam
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.n_epochs   = n_epochs
        self.batch_size = batch_size

        self.net = ActorCritic(obs_dim, n_actions)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    # --- Rollout storage ----------------------------------------------------

    def _gae(self, rewards, values, dones, last_value):
        """
        Generalised Advantage Estimation.
        Returns advantages and discounted returns.
        """
        T         = len(rewards)
        adv       = np.zeros(T, dtype=np.float32)
        last_gae  = 0.0
        values    = np.append(values, last_value)

        for t in reversed(range(T)):
            mask       = 1.0 - float(dones[t])
            delta      = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_gae   = delta + self.gamma * self.lam * mask * last_gae
            adv[t]     = last_gae

        returns = adv + values[:-1]
        return adv, returns

    # --- Update -------------------------------------------------------------

    def update(self, rollout):
        """
        Run n_epochs of minibatch PPO updates on a single collected rollout.
        rollout: dict with keys obs, actions, log_probs_old, advantages, returns.
        """
        obs        = rollout["obs"]        # (T,)  int
        actions    = rollout["actions"]    # (T,)  int
        lp_old     = rollout["log_probs"]  # (T,)  float
        advantages = rollout["advantages"] # (T,)  float
        returns    = rollout["returns"]    # (T,)  float

        T = len(obs)

        # One-hot encode the whole rollout at once
        obs_oh = torch.zeros(T, self.obs_dim)
        obs_oh[torch.arange(T), obs] = 1.0

        act_t   = torch.tensor(actions,    dtype=torch.long)
        lp_t    = torch.tensor(lp_old,     dtype=torch.float32)
        adv_t   = torch.tensor(advantages, dtype=torch.float32)
        ret_t   = torch.tensor(returns,    dtype=torch.float32)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            idx   = torch.randperm(T)
            for start in range(0, T, self.batch_size):
                mb = idx[start : start + self.batch_size]

                lp_new, vals, entropy = self.net.evaluate(obs_oh[mb], act_t[mb])

                ratio     = (lp_new - lp_t[mb]).exp()
                surr1     = ratio * adv_t[mb]
                surr2     = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.functional.mse_loss(vals, ret_t[mb])
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                total_loss += loss.item()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

        return total_loss

    # --- Collect one rollout ------------------------------------------------

    def collect_rollout(self, env, rollout_steps=512, max_ep_steps=1000):
        obs_buf, act_buf, lp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        obs  = env.reset()
        done = False
        ep_steps = 0

        for _ in range(rollout_steps):
            a, lp, v = self.net.act(obs, self.obs_dim)
            next_obs, r, done = env.step(a)

            obs_buf.append(obs)
            act_buf.append(a)
            lp_buf.append(lp)
            rew_buf.append(r)
            val_buf.append(v)
            done_buf.append(done)

            obs = next_obs
            ep_steps += 1

            if done or ep_steps >= max_ep_steps:
                obs  = env.reset()
                done = False
                ep_steps = 0

        # Bootstrap value at end of rollout
        _, _, last_val = self.net.act(obs, self.obs_dim)
        if done:
            last_val = 0.0

        adv, ret = self._gae(
            np.array(rew_buf, dtype=np.float32),
            np.array(val_buf, dtype=np.float32),
            np.array(done_buf, dtype=np.float32),
            last_val,
        )

        return {
            "obs":        np.array(obs_buf),
            "actions":    np.array(act_buf),
            "log_probs":  np.array(lp_buf, dtype=np.float32),
            "advantages": adv,
            "returns":    ret,
        }

    # --- Greedy inference ---------------------------------------------------

    def act_greedy(self, obs_idx):
        """Pick the highest-probability action (no sampling)."""
        x      = torch.zeros(self.obs_dim)
        x[obs_idx] = 1.0
        with torch.no_grad():
            logits, _ = self.net(x.unsqueeze(0))
        return logits.argmax().item()


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

def train(env, agent, n_updates=200, rollout_steps=512):
    episode_returns = []
    losses = []

    print("Training PPO agent...")
    for update in range(n_updates):
        rollout = agent.collect_rollout(env, rollout_steps=rollout_steps)
        loss    = agent.update(rollout)
        losses.append(loss)

        # Evaluate one greedy episode for logging
        obs  = env.reset()
        done = False
        ep_ret = 0.0
        for _ in range(1000):
            a    = agent.act_greedy(obs)
            obs, r, done = env.step(a)
            ep_ret += r
            if done:
                break
        episode_returns.append(ep_ret)

        if (update + 1) % 20 == 0:
            avg = np.mean(episode_returns[-20:])
            print(f"  Update {update+1:4d} | Avg return (last 20): {avg:7.2f} | Loss: {loss:.3f}")

    return episode_returns, losses


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Build the grid map
    grid_size = (12, 12)
    pg  = PipeGrid(*grid_size)
    vis = PipeVisualizerBW(lanes=2, base=3)
    bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))

    lane_coords  = [tuple(p) for p in np.argwhere(bw_map == 1)]
    start_pos    = lane_coords[0]
    package_pos  = lane_coords[len(lane_coords) // 2]
    delivery_pos = lane_coords[-1]

    # 2. Build env and agent
    env   = DeliveryEnv(bw_map, start_pos, package_pos, delivery_pos)
    agent = PPOAgent(
        obs_dim    = env.obs_dim,
        n_actions  = env.n_actions,
        lr         = 3e-4,
        gamma      = 0.99,
        lam        = 0.95,
        clip_eps   = 0.2,
        vf_coef    = 0.5,
        ent_coef   = 0.01,
        n_epochs   = 4,
        batch_size = 64,
    )

    # 3. Train
    returns, losses = train(env, agent, n_updates=200, rollout_steps=512)

    # 4. Plot training curve
    fig_train, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(returns);  ax1.set_title("Greedy episode return per update"); ax1.set_xlabel("Update")
    ax2.plot(losses);   ax2.set_title("PPO loss per update");              ax2.set_xlabel("Update")
    plt.tight_layout()
    plt.savefig("ppo_training.png")
    plt.show()

    # 5. Run a greedy inference episode and record the path
    obs      = env.reset()
    path     = [env.full_state()]
    done     = False
    print("\nRunning inference...")
    for step in range(1000):
        a        = agent.act_greedy(obs)
        obs, r, done = env.step(a)
        path.append(env.full_state())
        if done:
            print(f"Mission complete in {step+1} steps!")
            break
    else:
        print("Did not reach goal within 1000 steps.")

    # 6. Animate
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bw_map, cmap="gray")

    p_mark,    = ax.plot(package_pos[1],  package_pos[0],  "ys", markersize=10, label="Package")
    d_mark,    = ax.plot(delivery_pos[1], delivery_pos[0], "gx", markersize=12, label="Goal")
    drone_mark, = ax.plot([], [],                          "ro", markersize=7,  label="Drone")
    title = ax.set_title("Status: En Route to Package")

    def update(frame):
        r, c, has_pkg = path[frame]
        drone_mark.set_data([c], [r])
        if has_pkg:
            drone_mark.set_color("orange")
            p_mark.set_visible(False)
            title.set_text("Status: Delivering Package")
        else:
            drone_mark.set_color("red")
            p_mark.set_visible(True)
            title.set_text("Status: Fetching Package")
        return drone_mark, p_mark, title

    ani = animation.FuncAnimation(fig, update, frames=len(path), blit=False, interval=40)
    plt.legend(loc="lower right")
    ani.save("delivery_ppo.gif", writer="pillow")
    plt.show()