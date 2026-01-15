"""
HRL (Hierarchical Reinforcement Learning) on Robosuite
Understanding the Research Framework vs Using Pre-built Libraries
"""

print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║             HRL 算法研究 vs SB3库使用 - 完整思路梳理                            ║
╚════════════════════════════════════════════════════════════════════════════════╝


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ PART 1: 能否不关注 obs 处理直接使用 SB3?                                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

答案：可以 - 但取决于你的研究目标

├─ 如果你只是想 BENCHMARK（基准测试）
│  └─ ✓ 可以直接忽略 obs 处理细节
│  └─ SB3 PPO/SAC 的结果就是你的 baseline
│  └─ 但你需要确认 flatten_obs 的影响
│     (因为 obs 处理本身会影响学习效率)

├─ 如果你要研究自己的算法
│  └─ ✗ 需要关注 obs 处理
│  └─ 原因1：需要确保 "控制变量"
│     你的HRL算法 vs SB3 baseline 应该用相同的obs
│  └─ 原因2：需要理解模型输入
│     HRL 的不同层级可能需要不同 obs 粒度

└─ 总结：
   SB3库帮助：快速获得 baseline，省略实现细节
   你的研究：需要在相同的 obs 处理基础上进行
   不能混用：HRL用原始obs，baseline用flatten_obs → 不公平对比


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ PART 2: 研究算法 + 控制变量的最佳实践                                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

关键原则：所有对比实验都应该在相同的 "obs 处理管道" 上进行

标准研究框架：

┌──────────────────────────────────────────────────────────────┐
│  [robosuite 环境]                                            │
│       ↓                                                       │
│  [统一的 obs 处理层]  ← 关键！所有算法共用              │
│       ↓                                                       │
│  [算法选择]                                                  │
│       ├─ SB3 PPO (baseline)                                  │
│       ├─ SB3 SAC (baseline)                                  │
│       └─ 你的 HRL 算法 (新算法)                              │
│       ↓                                                       │
│  [相同的评估指标]                                             │
│       └─ 同样的环境、obs、reward                             │
└──────────────────────────────────────────────────────────────┘


示例：Lift 任务的公平对比设置

# 第1步：定义统一的 obs 处理配置
OBS_CONFIG = {
    "flatten_obs": True,      # 所有算法都用这个设置
    "keys": ["object-state", "robot0_proprio-state"],  # 显式指定
    "obs_dim": 60
}

# 第2步：创建相同的环境包装
def create_env():
    rs_env = suite.make(
        env_name="Lift",
        robots="Panda",
        use_camera_obs=False,
        reward_shaping=True,
        ...
    )
    return GymWrapper(
        rs_env,
        keys=OBS_CONFIG["keys"],
        flatten_obs=OBS_CONFIG["flatten_obs"]
    )

# 第3步：测试不同算法（在同一 obs 上）
## Baseline 1: SB3 PPO
env = create_env()
model_ppo = PPO("MlpPolicy", env, ...)
model_ppo.learn(100000)

## Baseline 2: SB3 SAC
env = create_env()
model_sac = SAC("MlpPolicy", env, ...)
model_sac.learn(100000)

## 你的算法: HRL
env = create_env()
model_hrl = YourHRLAlgorithm(obs_dim=OBS_CONFIG["obs_dim"], ...)
model_hrl.train(env, 100000)

# 第4步：相同条件评估
for model in [model_ppo, model_sac, model_hrl]:
    evaluate(model, create_env(), n_episodes=10, ...)


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ PART 3: HRL 算法的两种实现方式                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

╔═══════════════════════════════════════════════════════════════════════════════╗
║ 方式 A: 使用 robosuite 作为黑盒环境（推荐新手）                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

特点：
  ✓ robosuite 环境保持不变
  ✓ 只在上层实现 HRL 逻辑
  ✗ 层级之间的通信基于 "空间的划分"（低层做什么动作）
  ✗ 无法定制任务的中间目标状态

架构：

┌─────────────────────────────────────┐
│   High-level Policy (高层 HRL)      │
│  [目标: reach object, pick it]      │
│   ↓ 输出: 期望状态或子目标          │
├─────────────────────────────────────┤
│   Low-level Policy (低层，SB3)      │
│  [目标: 执行高层指令]               │
│   ↓ 输出: 动作                      │
├─────────────────────────────────────┤
│   robosuite 环境                    │
│   [执行动作, 返回 obs + reward]    │
└─────────────────────────────────────┘

代码示例：

class HighLevelPolicy:
    """High-level policy: task decomposition"""
    def __init__(self):
        self.state = "init"  # init -> reach -> grasp -> lift
    
    def get_subgoal(self, obs):
        """Return subgoal based on current observation"""
        object_pos = obs[0:3]  # 假设这是 object-state 的前3个
        gripper_pos = obs[10:13]  # 假设这是 robot state 中的 gripper pos
        
        if self.state == "init":
            return object_pos, "reach"
        elif self.state == "reach":
            distance = np.linalg.norm(object_pos - gripper_pos)
            if distance < 0.05:
                return object_pos, "grasp"
            return object_pos, "reach"
        # ... 更多状态

class HierarchicalAgent:
    def __init__(self, low_level_model):
        self.high_policy = HighLevelPolicy()
        self.low_policy = low_level_model  # SB3 模型
    
    def act(self, obs):
        # 获取子目标
        subgoal, phase = self.high_policy.get_subgoal(obs)
        
        # 增强 obs：加入子目标信息
        obs_augmented = np.concatenate([obs, subgoal])
        
        # 低层策略执行
        action = self.low_policy.predict(obs_augmented)[0]
        return action


╔═══════════════════════════════════════════════════════════════════════════════╗
║ 方式 B: 修改/扩展 robosuite 环境（高级，更灵活）                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

特点：
  ✓ 可以定义明确的"中间子目标"
  ✓ 可以为每个子任务设计 reward shaping
  ✓ 更接近学术研究中的标准 HRL 设置
  ✗ 需要更多的代码修改
  ✗ 需要理解 robosuite 内部结构

架构：

┌──────────────────────────────────────────────────────┐
│  修改后的 robosuite 环境                             │
│  ├─ 支持获取"子目标完成度"                           │
│  ├─ 支持中间 reward (到达子目标的 reward)           │
│  ├─ 支持 reset 到中间状态                           │
│  └─ 支持多粒度的 obs (完整obs vs 子任务obs)        │
│       ↓                                              │
│  高层策略 (选择子任务 t_1...t_n)                     │
│       ↓                                              │
│  低层策略 (在子任务 t_i 下执行, 最多 τ 步)          │
│       ↓                                              │
│  环境: 执行动作、返回 (obs, reward, subgoal_done)  │
└──────────────────────────────────────────────────────┘

代码示例：

class LiftEnvWithSubtasks:
    """Modified Lift environment with subtask support"""
    def __init__(self):
        self.env = suite.make(env_name="Lift", ...)
        self.subtasks = ["reach", "grasp", "lift"]
        self.current_subtask = None
    
    def set_subtask(self, subtask_name):
        """High-level policy tells env which subtask to do"""
        self.current_subtask = subtask_name
        self.subtask_step = 0
    
    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)
        
        # 计算子任务完成度
        subgoal_reward = self._compute_subgoal_reward(obs)
        
        # 环境返回扩展信息
        info["subtask"] = self.current_subtask
        info["subgoal_reward"] = subgoal_reward
        info["subgoal_done"] = subgoal_reward > 0.9
        
        return obs, env_reward, done, info
    
    def _compute_subgoal_reward(self, obs):
        """Compute intermediate reward based on subtask"""
        if self.current_subtask == "reach":
            # reach: get closer to object
            gripper_pos = obs[10:13]
            object_pos = obs[0:3]
            distance = np.linalg.norm(object_pos - gripper_pos)
            return 1.0 - min(distance / 0.5, 1.0)  # max distance 0.5m
        
        elif self.current_subtask == "grasp":
            # grasp: hold object
            gripper_state = obs[...]  # need to find correct index
            return gripper_state  # assume direct use
        
        elif self.current_subtask == "lift":
            # lift: object height increases
            object_height = obs[1]  # 假设 object_pos[1] 是高度
            return object_height
        
        return 0.0

class HierarchicalAgent:
    def __init__(self):
        self.env = LiftEnvWithSubtasks()
        self.high_policy = HighLevelPolicy()
        self.low_policies = {
            "reach": PPO.load("reach_policy"),
            "grasp": PPO.load("grasp_policy"),
            "lift": PPO.load("lift_policy"),
        }
    
    def train_episode(self):
        obs = self.env.reset()
        episode_reward = 0
        
        for high_step in range(10):  # max 10 subtasks
            # High-level policy selects subtask
            subtask = self.high_policy.select_task(obs)
            self.env.set_subtask(subtask)
            
            # Low-level policy executes for tau steps
            for low_step in range(50):  # 50 steps per subtask
                action = self.low_policies[subtask].predict(obs)[0]
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                
                # If subgoal done, move to next subtask
                if info["subgoal_done"]:
                    break
                
                # If environment done, entire task complete
                if done:
                    return episode_reward
        
        return episode_reward


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ PART 4: 对比分析 - 两种方式的优缺点                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┌────────────────────┬──────────────────────────────┬──────────────────────────────┐
│ 方面               │ 方式 A：黑盒 + 上层HRL      │ 方式 B：修改环境 + HRL      │
├────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 实现难度           │ ★★☆☆☆ (简单)              │ ★★★★★ (复杂)              │
│ 对 robosuite 改动  │ 无                          │ 大量修改                     │
│ 灵活性             │ ★★☆☆☆ (受限)              │ ★★★★★ (高)               │
│ 学术论文相关度     │ ★★☆☆☆ (创意性)            │ ★★★★★ (标准HRL)          │
│ 子目标定义清晰度   │ ★☆☆☆☆ (隐式)              │ ★★★★★ (显式)             │
│ 性能通常           │ ★★☆☆☆ (较低)              │ ★★★★☆ (较高)             │
│ 可复现性           │ ★★☆☆☆ (难重现)            │ ★★★★☆ (易重现)           │
└────────────────────┴──────────────────────────────┴──────────────────────────────┘


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ PART 5: 推荐研究路径                                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

第1阶段：理解 & Baseline（2-3周）
├─ ✓ 使用 SB3 PPO/SAC 在 Lift 上训练
├─ ✓ 理解观察空间的详细布局
├─ ✓ 获得 baseline 性能数据
├─ ✓ 建立标准的评估框架
└─ 输出：baseline.csv (训练曲线、最终性能)

第2阶段：HRL 原型（4-6周）
├─ 采用"方式 A"实现简单 HRL
├─ 在同样条件下对比 vs baseline
├─ 调试、优化 HRL 的超参数
└─ 输出：hrl_v1 vs baseline 的对比论文 / 报告

第3阶段：HRL 改进（6-8周）
├─ 转向"方式 B"，修改环境支持更复杂的子任务
├─ 实现更精细的奖励塑造
├─ 可能集成选项调用(Options Framework)或其他HRL技巧
└─ 输出：更好的性能、可发表的结果

第4阶段：多任务迁移（4周）
├─ 在多个 robosuite 任务上验证
├─ PickPlace, Assembly 等
├─ 证明 HRL 的泛化能力
└─ 输出：多任务结果、泛化能力分析


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ PART 6: 具体工程建议                                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

1. 项目文件结构
   
   your_project/
   ├── configs/
   │   ├── baseline.yaml      # SB3 PPO 参数
   │   ├── baseline_sac.yaml  # SB3 SAC 参数
   │   └── hrl.yaml           # 你的 HRL 参数
   │
   ├── envs/
   │   ├── robosuite_wrapper.py     # 统一的环境包装
   │   ├── lift_env_subtasks.py     # 改进的Lift环境(可选)
   │   └── obs_config.py            # 定义统一的obs处理
   │
   ├── algorithms/
   │   ├── baselines.py       # SB3 baseline 训练代码
   │   └── hrl.py             # 你的 HRL 实现
   │
   ├── experiments/
   │   ├── baseline_lift.py
   │   ├── hrl_lift.py
   │   └── eval.py            # 统一评估脚本
   │
   └── results/
       ├── baseline_ppo/
       ├── baseline_sac/
       └── hrl/

2. 统一的 obs 配置文件（全局参数）

   # obs_config.py
   ENV_CONFIG = {
       "Lift": {
           "flatten_obs": True,
           "keys": ["object-state", "robot0_proprio-state"],
           "obs_dim": 60,
           "subtasks": ["reach", "grasp", "lift"],
       },
       "PickPlace": {
           "flatten_obs": True,
           "keys": ["object-state", "robot0_proprio-state"],
           "obs_dim": 60,
           "subtasks": ["reach", "grasp", "place"],
       },
   }

3. 标准的训练脚本框架

   # train_baseline.py
   from configs import load_config
   from envs import make_env
   from stable_baselines3 import PPO
   
   config = load_config("configs/baseline.yaml")
   env = make_env("Lift", config["obs"])
   
   model = PPO(
       "MlpPolicy",
       env,
       learning_rate=config["lr"],
       n_steps=config["n_steps"],
       ...
   )
   model.learn(total_timesteps=config["total_steps"])
   model.save(config["save_path"])

4. 标准的评估框架

   # eval.py
   def evaluate_models(models, env, n_episodes=10):
       results = {}
       for model_name, model in models.items():
           returns = []
           for _ in range(n_episodes):
               obs = env.reset()
               total_return = 0
               while True:
                   action, _ = model.predict(obs, deterministic=True)
                   obs, reward, done, _ = env.step(action)
                   total_return += reward
                   if done:
                       returns.append(total_return)
                       break
           results[model_name] = {
               "mean": np.mean(returns),
               "std": np.std(returns),
               "returns": returns
           }
       return results


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ PART 7: 关键要点总结                                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

✓ 能否直接使用 SB3?
  └─ 可以，但要为 baseline。你的 HRL 应该在相同条件下测试。

✓ 如何保证公平对比?
  └─ 统一 obs 处理、统一环境、统一评估指标

✓ 如何实现 HRL?
  └─ 方式A (简单): 在 robosuite 上层堆砌 HRL 逻辑
  └─ 方式B (复杂): 修改 robosuite 环境支持子任务

✓ 研究路径?
  └─ Baseline → 简单HRL → 改进HRL → 多任务验证

✓ 不要做什么?
  ✗ 混合使用不同的 obs 处理 (baseline vs 你的HRL)
  ✗ 在不同的环境参数下对比
  ✗ 使用不同的评估指标
  ✗ 忽视"控制变量"原则

╔════════════════════════════════════════════════════════════════════════════════╗
║                              最后的建议                                        ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  1. 先用 SB3 获得 baseline (快速)                                             ║
║  2. 然后用"方式 A"实现简单 HRL (快速验证想法)                                 ║
║  3. 性能有改进后，再考虑转向"方式 B"做深入研究                                ║
║  4. 全程记录关键配置，便于复现和发表                                          ║
║                                                                                ║
║  This way you get:                                                            ║
║  ✓ 快速 baseline                                                              ║
║  ✓ 控制的变量                                                                ║
║  ✓ 公平的对比                                                                ║
║  ✓ 可复现的结果                                                              ║
║  ✓ 可发表的论文                                                              ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
""")
