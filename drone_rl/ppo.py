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
    Cell-level navigation on the PipeGrid (12x12 cells, not pixels).

    Valid moves are only the directions a cell is actually connected to
    (from PipeGrid.connections), so walls are naturally encoded.

    Observation (12-dim float):
      [r_norm, c_norm,
       goal_dr, goal_dc, goal_dist,
       pkg_dr,  pkg_dc,  pkg_dist,
       del_dr,  del_dc,  del_dist,
       has_pkg]
    """

    ACTIONS   = ["N", "S", "E", "W"]
    ACTION_DR = {"N": (-1,  0), "S": (1,  0), "E": (0,  1), "W": (0, -1)}

    def __init__(self, bw_map, start_pos, package_pos, delivery_pos,
                 connections, dir_map, grid_shape, max_steps=300):
        self.bw_map       = bw_map
        self.start_pos    = start_pos
        self.package_pos  = package_pos
        self.delivery_pos = delivery_pos
        self.connections  = connections   # list[list[set[str]]]
        self.rows, self.cols = grid_shape
        self.max_dist     = float(self.rows + self.cols)
        self.max_steps    = max_steps
        self.reset()

    # --- Observation --------------------------------------------------------

    def _obs(self, r, c, has_pkg):
        def vec(target):
            dr   = (target[0] - r) / self.max_dist
            dc   = (target[1] - c) / self.max_dist
            dist = (abs(target[0] - r) + abs(target[1] - c)) / self.max_dist
            return dr, dc, dist

        goal               = self.delivery_pos if has_pkg else self.package_pos
        g_dr, g_dc, g_dist = vec(goal)
        p_dr, p_dc, p_dist = vec(self.package_pos)
        d_dr, d_dc, d_dist = vec(self.delivery_pos)

        return np.array([
            r / self.rows, c / self.cols,
            g_dr, g_dc, g_dist,
            p_dr, p_dc, p_dist,
            d_dr, d_dc, d_dist,
            float(has_pkg),
        ], dtype=np.float32)

    @property
    def obs_dim(self):   return 12

    @property
    def n_actions(self): return len(self.ACTIONS)

    # --- Core API -----------------------------------------------------------

    def reset(self):
        r, c       = self.start_pos
        self.state = (r, c, False)
        self.steps = 0
        return self._obs(r, c, False)

    def step(self, action):
        r, c, has_pkg = self.state
        direction     = self.ACTIONS[action]
        dr, dc        = self.ACTION_DR[direction]

        # Only move if this direction is a valid pipe connection
        if direction in self.connections[r][c]:
            nr, nc = r + dr, c + dc
        else:
            nr, nc = r, c  # blocked — stay in place

        self.steps += 1
        reward = -0.05  # time penalty

        # Potential shaping toward current goal
        goal      = self.delivery_pos if has_pkg else self.package_pos
        prev_dist = abs(r  - goal[0]) + abs(c  - goal[1])
        new_dist  = abs(nr - goal[0]) + abs(nc - goal[1])
        reward   += 0.2 * (prev_dist - new_dist)

        if not has_pkg and (nr, nc) == self.package_pos:
            has_pkg = True
            reward += 5.0

        done    = False
        success = False
        if has_pkg and (nr, nc) == self.delivery_pos:
            reward += 50.0
            done    = True
            success = True

        if self.steps >= self.max_steps:
            done = True

        self.state = (nr, nc, has_pkg)
        return self._obs(nr, nc, has_pkg), reward, done, success

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

    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs_vec):
        """Sample an action. obs_vec: np.ndarray shape (obs_dim,)"""
        x = torch.from_numpy(obs_vec).unsqueeze(0)
        with torch.no_grad():
            logits, value = self(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs_batch, act_batch):
        """Evaluate a batch of (obs, action) pairs for the PPO update."""
        logits, values = self(obs_batch)
        dist      = Categorical(logits=logits)
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
        lr         = 3e-4,
        gamma      = 0.99,
        lam        = 0.95,
        clip_eps   = 0.2,
        vf_coef    = 0.5,
        ent_coef   = 0.001,
        n_epochs   = 4,
        batch_size = 64,
    ):
        self.gamma      = gamma
        self.lam        = lam
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.n_epochs   = n_epochs
        self.batch_size = batch_size

        self.net = ActorCritic(obs_dim, n_actions)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        # Running stats for return normalisation (keeps value targets ~unit scale)
        self._ret_mean = 0.0
        self._ret_var  = 1.0
        self._ret_count = 0

    # --- GAE ----------------------------------------------------------------

    def _gae(self, rewards, values, dones, last_value):
        T        = len(rewards)
        adv      = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        values   = np.append(values, last_value)

        for t in reversed(range(T)):
            mask     = 1.0 - float(dones[t])
            delta    = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_gae = delta + self.gamma * self.lam * mask * last_gae
            adv[t]   = last_gae

        return adv, adv + values[:-1]

    # --- Update -------------------------------------------------------------

    def update(self, rollout):
        obs        = torch.tensor(rollout["obs"],        dtype=torch.float32)
        act_t      = torch.tensor(rollout["actions"],    dtype=torch.long)
        lp_t       = torch.tensor(rollout["log_probs"],  dtype=torch.float32)
        adv_t      = torch.tensor(rollout["advantages"], dtype=torch.float32)
        ret_raw    = rollout["returns"]

        # Update running return statistics and normalise
        batch_mean = ret_raw.mean()
        batch_var  = ret_raw.var() + 1e-8
        self._ret_count += 1
        self._ret_mean  += (batch_mean - self._ret_mean) / self._ret_count
        self._ret_var    = 0.99 * self._ret_var + 0.01 * batch_var
        ret_norm = (ret_raw - self._ret_mean) / (np.sqrt(self._ret_var) + 1e-8)
        ret_t    = torch.tensor(ret_norm, dtype=torch.float32)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        T          = len(obs)
        total_loss = 0.0

        for _ in range(self.n_epochs):
            idx = torch.randperm(T)
            for start in range(0, T, self.batch_size):
                mb = idx[start : start + self.batch_size]

                lp_new, vals, entropy = self.net.evaluate(obs[mb], act_t[mb])

                ratio    = (lp_new - lp_t[mb]).exp()
                surr1    = ratio * adv_t[mb]
                surr2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]

                loss = (
                    -torch.min(surr1, surr2).mean()
                    + self.vf_coef  * nn.functional.mse_loss(vals, ret_t[mb])
                    - self.ent_coef * entropy.mean()
                )
                total_loss += loss.item()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

        return total_loss

    # --- Collect one rollout ------------------------------------------------

    def collect_rollout(self, env, rollout_steps=2048, max_ep_steps=500):
        obs_buf, act_buf, lp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        obs  = env.reset()
        done = False

        for _ in range(rollout_steps):
            a, lp, v          = self.net.act(obs)
            next_obs, r, done, _ = env.step(a)

            obs_buf.append(obs)
            act_buf.append(a)
            lp_buf.append(lp)
            rew_buf.append(r)
            val_buf.append(v)
            done_buf.append(done)

            obs = next_obs
            if done:
                obs  = env.reset()
                done = False

        # Bootstrap
        with torch.no_grad():
            _, last_val = self.net(torch.from_numpy(obs).unsqueeze(0))
        last_val = 0.0 if done else last_val.item()

        adv, ret = self._gae(
            np.array(rew_buf,  dtype=np.float32),
            np.array(val_buf,  dtype=np.float32),
            np.array(done_buf, dtype=np.float32),
            last_val,
        )

        return {
            "obs":        np.array(obs_buf,  dtype=np.float32),
            "actions":    np.array(act_buf),
            "log_probs":  np.array(lp_buf,   dtype=np.float32),
            "advantages": adv,
            "returns":    ret,
        }

    # --- Greedy inference ---------------------------------------------------

    def act_greedy(self, obs_vec):
        x = torch.from_numpy(obs_vec).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
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
        obs    = env.reset()
        done   = False
        ep_ret = 0.0
        for _ in range(500):
            a                    = agent.act_greedy(obs)
            obs, r, done, success = env.step(a)
            ep_ret += r
            if done:
                break
        episode_returns.append(ep_ret)

        if (update + 1) % 20 == 0:
            avg    = np.mean(episode_returns[-20:])
            status = "SUCCESS" if success else "timeout"
            print(f"  Update {update+1:4d} | Avg return (last 20): {avg:7.2f} | Loss: {loss:.3f} | {status}")

    return episode_returns, losses


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Build the grid at CELL level (12x12)
    #    We use the PipeGrid connection graph directly — each cell is a node,
    #    and valid moves are only the directions that cell is connected to.
    #    This avoids navigating a pixel maze and keeps the state space small.
    grid_size = (12, 12)
    pg        = PipeGrid(*grid_size)

    # Build a per-cell adjacency: cell (r,c) -> set of reachable neighbor cells
    connections = pg.connections  # list[list[set[str]]]

    dir_map = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

    rows, cols = grid_size
    all_cells  = [(r, c) for r in range(rows) for c in range(cols)]

    # Pick start, package, delivery as spread-out cells
    start_pos    = (0, 0)
    package_pos  = (rows // 2, cols // 2)
    delivery_pos = (rows - 1, cols - 1)

    # Build bw_map only for visualization
    vis    = PipeVisualizerBW(lanes=2, base=3)
    bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))

    # 2. Build env and agent — pass connections so env moves at cell level
    env   = DeliveryEnv(
        bw_map, start_pos, package_pos, delivery_pos,
        connections=connections, dir_map=dir_map, grid_shape=(rows, cols),
    )
    agent = PPOAgent(
        obs_dim    = env.obs_dim,
        n_actions  = env.n_actions,
        lr         = 3e-4,
        gamma      = 0.99,
        lam        = 0.95,
        clip_eps   = 0.2,
        vf_coef    = 0.5,
        ent_coef   = 0.001,
        n_epochs   = 4,
        batch_size = 64,
    )

    # 3. Train
    returns, losses = train(env, agent, n_updates=500, rollout_steps=2048)

    # 4. Plot training curve
    fig_train, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(returns);  ax1.set_title("Greedy episode return per update"); ax1.set_xlabel("Update")
    ax2.plot(losses);   ax2.set_title("PPO loss per update");              ax2.set_xlabel("Update")
    plt.tight_layout()
    plt.savefig("ppo_training.png")
    plt.show()

    # 5. Run a greedy inference episode and record the path
    obs     = env.reset()
    path    = [env.full_state()]
    success = False
    print("\nRunning inference...")
    for step in range(500):
        a                     = agent.act_greedy(obs)
        obs, r, done, success = env.step(a)
        path.append(env.full_state())
        if done:
            break

    if success:
        print(f"Mission complete in {step+1} steps!")
    else:
        print(f"Did not reach goal (timed out after {step+1} steps). "
              f"Final state: {env.full_state()}")

    # 6. Animate — map cell positions back to pixel centres for display
    cell_px = vis.s  # pixels per cell = lanes * base = 6
    def cell_to_px(r, c):
        return c * cell_px + cell_px // 2, r * cell_px + cell_px // 2  # x, y

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bw_map, cmap="gray")

    px, py           = cell_to_px(*package_pos)
    dx, dy           = cell_to_px(*delivery_pos)
    p_mark,          = ax.plot(px, py, "ys", markersize=10, label="Package")
    d_mark,          = ax.plot(dx, dy, "gx", markersize=12, label="Goal")
    drone_mark,      = ax.plot([], [],  "ro", markersize=7,  label="Drone")
    title             = ax.set_title("Status: En Route to Package")

    def update(frame):
        r, c, has_pkg = path[frame]
        cx, cy = cell_to_px(r, c)  # cx=col pixels, cy=row pixels
        drone_mark.set_data([cx], [cy])  # plot takes (x=col, y=row)
        if has_pkg:
            drone_mark.set_color("orange")
            p_mark.set_visible(False)
            title.set_text("Status: Delivering Package")
        else:
            drone_mark.set_color("red")
            p_mark.set_visible(True)
            title.set_text("Status: Fetching Package")
        return drone_mark, p_mark, title

    ani = animation.FuncAnimation(fig, update, frames=len(path), blit=False, interval=80)
    plt.legend(loc="lower right")
    ani.save("delivery_ppo.gif", writer="pillow")
    plt.show()