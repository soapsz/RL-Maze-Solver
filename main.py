import logging
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np

import models
import mazesetup
from environment.maze import Maze, Render

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)  # Only show messages *equal to or above* this level


class Test(Enum):
    SHOW_MAZE_ONLY = auto()
    VALUE_ITERATION = auto()
    Q_LEARNING = auto()
    Q_ELIGIBILITY = auto()
    SARSA = auto()
    SARSA_ELIGIBILITY = auto()
    # -- Newly added test cases
    DYNA_Q = auto()
    Q_LEARNING_COMPARISON = auto()
    DYNA_Q_VS_QL_STATIC = auto() # For Dyna-Q comparison with Q-Learning with static maze
    DYNA_Q_VS_QL_DYNAMIC = auto() # For Dyna-Q comparison with Q-Learning with dynamic maze
    # ALL_MODELS_DYNAMIC_TEST = auto() # <-- This is now REMOVED
    
    # --- New individual dynamic tests ---
    Q_LEARNING_DYNAMIC = auto()
    DYNA_Q_DYNAMIC = auto()
    SARSA_DYNAMIC = auto()
    Q_TRACE_DYNAMIC = auto()
    SARSA_TRACE_DYNAMIC = auto()

test = Test.SARSA_TRACE_DYNAMIC # which test to run

mazeType = "blank18" # maze types

if mazeType == "normal8":
    maze = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0]
    ])  # 0 = free, 1 = occupied
elif mazeType == "blank8":
    maze = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])  # 0 = free, 1 = occupied
elif mazeType == "normal18":
    maze = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
elif mazeType == "blank18":
    maze = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])  # 0 = free, 1 = occupied



## ----- Maze setup use for Dyna-Q comparison, I moved mazes to new file mazesetup.py 
#  ----- to reduce clutter
comparison_maze = mazesetup.comparison_maze
comparison_maze_dynamic = mazesetup.comparison_maze_dynamic

game = Maze(maze)

## -- New section added
comparison_game1 = Maze(comparison_maze)
dynamic_game1 = Maze(comparison_maze)
dynamic_game2 = Maze(comparison_maze_dynamic)

# only show the maze
if test == Test.SHOW_MAZE_ONLY:
    game.render(Render.MOVES)
    game.reset()

# plan using value iteration
if test == Test.VALUE_ITERATION:
    game.render(Render.TRAINING)
    model = models.ValueIterationModel(game)
    h, w, _, _ = model.train(discount=0.90, theta=1e-4, max_iterations=200)

# train using tabular Q-learning
if test == Test.Q_LEARNING:
    game.render(Render.TRAINING)
    model = models.QTableModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)

# train using tabular Q-learning and an eligibility trace (aka TD-lambda)
if test == Test.Q_ELIGIBILITY:
    game.render(Render.TRAINING)
    model = models.QTableTraceModel(game)
    h, w, _, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)

# train using tabular SARSA learning
if test == Test.SARSA:
    game.render(Render.TRAINING)
    model = models.SarsaTableModel(game)
    h, w, _, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)

# train using tabular SARSA learning and an eligibility trace
if test == Test.SARSA_ELIGIBILITY:
    game.render(Render.TRAINING)  # shows all moves and the q table; nice but slow.
    model = models.SarsaTableTraceModel(game)
    h, w, _, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                             stop_at_convergence=True)
    
## --- NEW TEST SETUP SECTION HERE

# Static Maze setup for Dyna-Q vs Q-Learning
if test == Test.DYNA_Q_VS_QL_STATIC:
    

    q_model = models.QTable2CModel(comparison_game1)
    dq_model = models.DynaQModel(comparison_game1)

    comparison_game1.render(Render.NOTHING)
    h1, w1, _, _, q_metrics = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=300, stop_at_convergence=False
    )

    print("Showing learned path for DQ_Maze1 (Q-learning)...")
    maze_q = Maze(comparison_game1.maze.copy())
    maze_q.render(Render.MOVES)
    maze_q.play(q_model, start_cell=(0, 0))

    comparison_game1.render(Render.NOTHING)
    h2, w2, _, _, dq_metrics = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=300, stop_at_convergence=False
    )
    print("Showing learned path for DQ_Maze1 (Dyna-Q)...")
    maze_dq = Maze(comparison_game1.maze.copy())
    maze_dq.render(Render.MOVES)
    maze_dq.play(dq_model, start_cell=(0, 0))

    # Plot metrics
    q_episodes = [m["episode"] for m in q_metrics]
    q_returns = [m["return_"] for m in q_metrics]
    q_steps = [m["steps"] for m in q_metrics]
    q_success = [m["success"] for m in q_metrics]
    q_cumulative_steps = [m["cumulative_steps"] for m in q_metrics]

    dq_episodes = [m["episode"] for m in dq_metrics]
    dq_returns = [m["return_"] for m in dq_metrics]
    dq_steps = [m["steps"] for m in dq_metrics]
    dq_success = [m["success"] for m in dq_metrics]
    dq_cumulative_steps = [m["cumulative_steps"] for m in dq_metrics]
    dq_cumulative_total_updates = [m["cumulative_total_updates"] for m in dq_metrics]

    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_returns, label="Total Rewards for Q-Learning")
    plt.plot(dq_episodes, dq_returns, label="Total Rewards for Dyna-Q")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-Learning vs Dyna-Q comparison: Total Rewards")
    plt.legend()

    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_steps, label="Steps Taken for Q-Learning")
    plt.plot(dq_episodes, dq_steps, label="Steps Taken for Dyna-Q")
    plt.xlabel("Episode")
    plt.ylabel("Steps Taken")
    plt.title("Q-Learning vs Dyna-Q comparison: Steps Taken")
    plt.legend()

    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    axes[0].plot(dq_episodes, dq_cumulative_steps, label="Exploration Steps")
    axes[0].set_title("Dyna-Q: Exploration Steps")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Steps")
    axes[0].legend()
    axes[1].plot(dq_episodes, dq_cumulative_total_updates, label="Total Steps (Exploration + Simulated)")
    axes[1].set_title("Dyna-Q: Total Steps: Exploration + Simulated")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Cumulative Total Updates")
    axes[1].legend()

    plt.show()


# Dynamic Maze setup for Dyna-Q vs Q-learning
if test == Test.DYNA_Q_VS_QL_DYNAMIC:
    q_model = models.QTable2CModel(Maze(dynamic_game1.maze.copy()))
    dq_model = models.DynaQModel(Maze(dynamic_game1.maze.copy()))
    
    episodes_before_change = 150

    dynamic_game1.render(Render.NOTHING)
    h1, w1, _, _, q_metrics_before = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False
    )
    print("Showing learned path before Change (Q-learning)...")
    maze_q = Maze(dynamic_game1.maze.copy())
    maze_q.render(Render.MOVES)
    maze_q.play(q_model, start_cell=(0, 0))

    dq_model.environment.render(Render.NOTHING)
    h2, w2, _, _, dq_metrics_before = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30
    )
    print("Showing learned path before Change (Dyna-Q)...")
    maze_dq = Maze(dynamic_game1.maze.copy())
    maze_dq.render(Render.MOVES)
    maze_dq.play(dq_model, start_cell=(0, 0))

    # Test VI
    vi_model_static = models.ValueIterationModel(Maze(dynamic_game1.maze.copy()))
    vi_model_static.train(discount=0.9)
    print("Showing optimal path (Value Iteration) - Original Maze:")
    maze_vi1 = Maze(dynamic_game1.maze.copy())
    maze_vi1.render(Render.MOVES)
    maze_vi1.play(vi_model_static, start_cell=(0, 0))
    

    # Change Maze Environment ------------------------------------------
    q_model.environment = Maze(comparison_maze_dynamic.copy())
    dq_model.environment = Maze(comparison_maze_dynamic.copy())

    h3, w3, _, _, q_metrics_after = q_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False
    )
    print("Showing learned path after Change (Q-learning)...")
    q_model.environment.render(Render.MOVES)
    q_model.environment.play(q_model, start_cell=(0, 0))

    h4, w4, _, _, dq_metrics_after = dq_model.train(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False, n_planning=30
    )
    print("Showing learned path after Change (Dyna-Q)...")
    dq_model.environment.render(Render.MOVES)
    dq_model.environment.play(dq_model, start_cell=(0, 0))

    # VI CHANGED
    vi_model_dynamic = models.ValueIterationModel(Maze(comparison_maze_dynamic.copy()))
    vi_model_dynamic.train(discount=0.9)
    print("Showing optimal path (Value Iteration) - Changed Maze:")
    maze_vi_fail = Maze(comparison_maze_dynamic.copy())
    maze_vi_fail.render(Render.MOVES)
    maze_vi_fail.play(vi_model_static, start_cell=(0, 0))

    for m in q_metrics_after:
        m["episode"] += episodes_before_change
    for m in dq_metrics_after:
        m["episode"] += episodes_before_change
    q_metrics = q_metrics_before + q_metrics_after
    dq_metrics = dq_metrics_before + dq_metrics_after



    # Plot metrics
    q_episodes = [m["episode"] for m in q_metrics]
    q_returns = [m["return_"] for m in q_metrics]
    q_steps = [m["steps"] for m in q_metrics]
    q_success = [m["success"] for m in q_metrics]
    q_cumulative_steps = [m["cumulative_steps"] for m in q_metrics]

    dq_episodes = [m["episode"] for m in dq_metrics]
    dq_returns = [m["return_"] for m in dq_metrics]
    dq_steps = [m["steps"] for m in dq_metrics]
    dq_success = [m["success"] for m in dq_metrics]
    dq_cumulative_steps = [m["cumulative_steps"] for m in dq_metrics]
    dq_cumulative_total_updates = [m["cumulative_total_updates"] for m in dq_metrics]

    def roll(y, w=25):
        y = np.asarray(y, float)
        return np.convolve(y, np.ones(w)/w, mode='valid')
    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_returns, label="Total Rewards for Q-Learning")
    plt.plot(dq_episodes, dq_returns, label="Total Rewards for Dyna-Q")
    plt.plot(q_episodes[len(q_episodes)-len(roll(q_returns)):], roll(q_returns), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(roll(dq_returns)):], roll(dq_returns), label="Dyna-Q (smoothed)")
    plt.axvline(episodes_before_change, color='red', linestyle='--', label="Maze Changed")
    plt.text(episodes_before_change+10, np.min(q_returns), "Maze Change", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-Learning vs Dyna-Q comparison: Total Rewards")
    plt.legend()

    

    plt.figure(figsize=(8,6))
    plt.plot(q_episodes, q_steps, alpha=0.35, label="Steps Q-Learning (raw)")
    plt.plot(dq_episodes, dq_steps, alpha=0.35, label="Steps Dyna-Q (raw)")
    plt.plot(q_episodes[len(q_episodes)-len(roll(q_steps)):], roll(q_steps), label="Q-Learning (smoothed)")
    plt.plot(dq_episodes[len(dq_episodes)-len(roll(dq_steps)):], roll(dq_steps), label="Dyna-Q (smoothed)")
    plt.axvline(episodes_before_change, color='red', ls='--', label="Maze Changed")
    plt.xlabel("Episode"); plt.ylabel("Steps Taken"); plt.title("Q-Learning vs Dyna-Q: Steps")
    plt.legend()

    plt.show()

    # def recovery_episodes(returns, t_change, w=25, frac=0.95):
    #     base = np.mean(returns[max(0,t_change-w):t_change])
    #     after = roll(returns[t_change:], w)
    #     meet = np.where(after >= frac*base)[0]
    #     return None if len(meet)==0 else int(meet[0])

    # rec_q  = recovery_episodes(q_returns,  episodes_before_change)
    # rec_dq = recovery_episodes(dq_returns, episodes_before_change)
    # print("Episodes to 95% recovery  |  Q-Learning:", rec_q, "  Dyna-Q:", rec_dq)


# --- Helper function for individual dynamic tests ---
def run_dynamic_test(model_class, model_name, **kwargs):
    
    # --- Model Initialization ---
    model = model_class(Maze(dynamic_game1.maze.copy()))
    
    metrics_before = {}
    metrics_after = {}
    
    episodes_before_change = 150
    episodes_after_change = 150
    n_planning = kwargs.get("n_planning", 30) # For Dyna-Q

    # --- Phase 1: Train on static maze ---
    print(f"--- Phase 1: Training {model_name} on Original Maze ---")
    model.environment.render(Render.NOTHING)
    
    train_kwargs = dict(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_before_change, stop_at_convergence=False
    )
    if model_name == "Dyna-Q":
        train_kwargs["n_planning"] = n_planning
        
    _, _, _, _, metrics_before = model.train(**train_kwargs)

    # --- Benchmark: Value Iteration (Static) ---
    vi_model_static = models.ValueIterationModel(Maze(dynamic_game1.maze.copy()))
    vi_model_static.train(discount=0.9)
    print("Showing optimal path (Value Iteration) - Original Maze:")
    maze_vi1 = Maze(dynamic_game1.maze.copy())
    maze_vi1.render(Render.MOVES)
    maze_vi1.play(vi_model_static, start_cell=(0, 0))
    

    # --- Phase 2: Change Maze Environment ---
    print(f"\n--- Phase 2: Changing Maze Environment for {model_name} ---")
    model.environment = Maze(comparison_maze_dynamic.copy())

    # --- Phase 3: Train on dynamic maze ---
    print(f"\n--- Phase 3: Continuing training for {model_name} on Changed Maze ---")
    
    train_kwargs = dict(
        discount=0.9, exploration_rate=0.1, learning_rate=0.1,
        episodes=episodes_after_change, stop_at_convergence=False
    )
    if model_name == "Dyna-Q":
        train_kwargs["n_planning"] = n_planning

    _, _, _, _, metrics_after = model.train(**train_kwargs)
    
    # --- [FIXED BLOCK] Show the model's path on the new maze ---
    print(f"\n--- Showing {model_name}'s learned path on Changed Maze ---")
    # Use the model's own environment, which it was just trained on
    model.environment.render(Render.MOVES)
    model.environment.play(model, start_cell=(0, 0))
    # --- End of fix ---

    # --- Benchmark: Value Iteration (Dynamic) ---
    vi_model_dynamic = models.ValueIterationModel(Maze(comparison_maze_dynamic.copy()))
    vi_model_dynamic.train(discount=0.9)
    print("Showing optimal path (Value Iteration) - Changed Maze:")
    maze_vi_dynamic = Maze(comparison_maze_dynamic.copy())
    maze_vi_dynamic.render(Render.MOVES)
    maze_vi_dynamic.play(vi_model_dynamic, start_cell=(0, 0))
    
    print("Showing static VI model failure on Changed Maze:")
    maze_vi_fail = Maze(comparison_maze_dynamic.copy())
    maze_vi_fail.render(Render.MOVES)
    maze_vi_fail.play(vi_model_static, start_cell=(0, 0)) # Show static model failing

    # --- Phase 4: Process and Plot Metrics ---
    for m in metrics_after:
        m["episode"] += episodes_before_change
    metrics_all = metrics_before + metrics_after

    # Helper for smoothing
    def roll(y, w=25):
        if len(y) < w:
            return y
        y = np.asarray(y, float)
        return np.convolve(y, np.ones(w)/w, mode='valid')

    # Plot Returns
    plt.figure(figsize=(12, 8))
    episodes = [m["episode"] for m in metrics_all]
    returns = [m["return_"] for m in metrics_all]
    
    # Plot smoothed data
    smoothed_returns = roll(returns)
    start_idx = len(episodes) - len(smoothed_returns)
    plt.plot(episodes[start_idx:], smoothed_returns, label=f"{model_name} (smoothed)")
    plt.plot(episodes, returns, label=f"{model_name} (raw)", alpha=0.3) # Add raw plot

    plt.axvline(episodes_before_change, color='red', linestyle='--', label="Maze Changed")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"{model_name}: Total Rewards on Dynamic Maze")
    plt.legend()

    # Plot Steps
    plt.figure(figsize=(12, 8))
    steps = [m["steps"] for m in metrics_all]

    # Plot smoothed data
    smoothed_steps = roll(steps)
    start_idx = len(episodes) - len(smoothed_steps)
    plt.plot(episodes[start_idx:], smoothed_steps, label=f"{model_name} (smoothed)")
    plt.plot(episodes, steps, label=f"{model_name} (raw)", alpha=0.3) # Add raw plot

    plt.axvline(episodes_before_change, color='red', linestyle='--', label="Maze Changed")
    plt.xlabel("Episode")
    plt.ylabel("Steps Taken")
    plt.title(f"{model_name}: Steps Taken on Dynamic Maze")
    plt.legend()
    
    plt.show()

# --- NEW: Individual Dynamic Test Blocks ---

if test == Test.Q_LEARNING_DYNAMIC:
    run_dynamic_test(models.QTable2CModel, "Q-Learning")

if test == Test.DYNA_Q_DYNAMIC:
    run_dynamic_test(models.DynaQModel, "Dyna-Q", n_planning=30)

if test == Test.SARSA_DYNAMIC:
    run_dynamic_test(models.SarsaTableModel, "SARSA")

if test == Test.Q_TRACE_DYNAMIC:
    run_dynamic_test(models.QTableTraceModel, "Q-Learning (Trace)")

if test == Test.SARSA_TRACE_DYNAMIC:
    run_dynamic_test(models.SarsaTableTraceModel, "SARSA (Trace)")


# Dynamic Maze setup for ALL models (REMOVED)
# if test == Test.ALL_MODELS_DYNAMIC_TEST:
#     ... (code block removed) ...


## --------------------------------

# List of all comparison tests that handle their own plotting
comparison_tests = [
    Test.DYNA_Q_VS_QL_STATIC, 
    Test.DYNA_Q_VS_QL_DYNAMIC, 
    Test.Q_LEARNING_DYNAMIC,
    Test.DYNA_Q_DYNAMIC,
    Test.SARSA_DYNAMIC,
    Test.Q_TRACE_DYNAMIC,
    Test.SARSA_TRACE_DYNAMIC
]

# draw graphs showing development of win rate and cumulative rewards
try:
    h  # force a NameError exception if h does not exist, and thus don't try to show win rate and cumulative reward
    
    # Do not show the default plot if running one of the comparison tests
    if test not in comparison_tests:
        fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
        fig.canvas.manager.set_window_title(model.name)
        if w:
            ax1.plot(*zip(*w))
            ax1.set_xlabel("episode")
            ax1.set_ylabel("win rate")
        else:
            ax1.set_axis_off()
            ax1.text(0.5, 0.5, "win rate unavailable", ha="center", va="center", transform=ax1.transAxes)
        ax2.plot(h)
        ax2.set_xlabel("episode")
        ax2.set_ylabel("cumulative reward")
        plt.show()
except NameError:
    pass

# Do not run the final "play" if running a comparison test
if test not in comparison_tests:
    try:
        game.render(Render.MOVES)
        game.play(model, start_cell=(4, 1))
    except NameError:
        pass # model was not defined, e.g. for SHOW_MAZE_ONLY

plt.show()  # must be placed here else the image disappears immediately at the end of the program