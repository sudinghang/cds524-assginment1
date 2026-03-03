import numpy as np
import random
import pickle
import time
import matplotlib.pyplot as plt

# -------------------------- 游戏核心参数配置 --------------------------
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600

PLAYER_SIZE = 30
PLAYER_SPEED = 5
ENEMY_SIZE = 25
ENEMY_SPEED = 2
BULLET_SIZE = 5
BULLET_SPEED = 15

# -------------------------- Q-Learning 参数 --------------------------
ACTIONS = [0, 1, 2, 3, 4]
ACTION_DELTA = [
    (0, -PLAYER_SPEED),  # 0=上(W)
    (0, PLAYER_SPEED),  # 1=下(S)
    (-PLAYER_SPEED, 0),  # 2=左(A)
    (PLAYER_SPEED, 0),  # 3=右(D)
    (0, 0)  # 4=射击(空格)
]

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
ALPHA = 0.2
GAMMA = 0.9

REWARD_HIT_ENEMY = 150
REWARD_COLLIDE = -300
REWARD_SHOOT = -1
REWARD_SURVIVE = 5
REWARD_OUT_BOUND = -20
REWARD_KILL_ALL = 1000

MAX_ENEMIES = 3
MAX_BULLETS = 3


# -------------------------- 游戏环境类（无图形渲染） --------------------------
class AirShooterEnv:
    def __init__(self):
        self.player_x = WINDOW_WIDTH // 2 - PLAYER_SIZE // 2
        self.player_y = WINDOW_HEIGHT - PLAYER_SIZE - 10
        self.player_live = True

        self.enemies = []
        self._spawn_enemies(MAX_ENEMIES)
        self.bullets = []

        self.kill_count = 0
        self.total_reward = 0

    def _spawn_enemies(self, num):
        for _ in range(num):
            while True:
                enemy_x = random.randint(ENEMY_SIZE, WINDOW_WIDTH - ENEMY_SIZE)
                enemy_y = random.randint(-100, -ENEMY_SIZE)
                enemy = (enemy_x, enemy_y)
                if not any([abs(enemy[0] - e[0]) < ENEMY_SIZE and abs(enemy[1] - e[1]) < ENEMY_SIZE for e in
                            self.enemies]):
                    self.enemies.append(enemy)
                    break

    def _is_out_bound(self, x, y, size):
        return x < 0 or x > WINDOW_WIDTH - size or y < 0 or y > WINDOW_HEIGHT - size

    def _check_collision(self, obj1, obj1_size, obj2, obj2_size):
        x1, y1 = obj1
        x2, y2 = obj2
        return (x1 < x2 + obj2_size and x1 + obj1_size > x2 and
                y1 < y2 + obj2_size and y1 + obj1_size > y2)

    def reset(self):
        self.__init__()
        return self._get_state()

    def _get_state(self):
        player_x_seg = int(self.player_x / (WINDOW_WIDTH / 10))
        player_y_seg = int(self.player_y / (WINDOW_HEIGHT / 10))

        if self.enemies:
            nearest_enemy = min(self.enemies,
                                key=lambda e: np.sqrt((e[0] - self.player_x) ** 2 + (e[1] - self.player_y) ** 2))
            enemy_x_seg = int(nearest_enemy[0] / (WINDOW_WIDTH / 10))
            enemy_y_seg = int(nearest_enemy[1] / (WINDOW_HEIGHT / 10))
        else:
            enemy_x_seg, enemy_y_seg = 0, 0

        bullet_exist = 1 if self.bullets else 0
        return (player_x_seg, player_y_seg, enemy_x_seg, enemy_y_seg, bullet_exist)

    def step(self, action):
        reward = 0
        done = False

        dx, dy = ACTION_DELTA[action]
        if action in [0, 1, 2, 3]:
            new_x = self.player_x + dx
            new_y = self.player_y + dy
            if not self._is_out_bound(new_x, new_y, PLAYER_SIZE):
                self.player_x = new_x
                self.player_y = new_y
            else:
                reward += REWARD_OUT_BOUND
        elif action == 4:
            if len(self.bullets) < MAX_BULLETS:
                bullet_x = self.player_x + PLAYER_SIZE // 2 - BULLET_SIZE // 2
                bullet_y = self.player_y - BULLET_SIZE
                self.bullets.append((bullet_x, bullet_y))
                reward += REWARD_SHOOT

        # 子弹逻辑
        new_bullets = []
        for bullet in self.bullets:
            bx, by = bullet
            by -= BULLET_SPEED
            if by > 0:
                new_bullets.append((bx, by))
                hit_enemies = []
                for enemy in self.enemies:
                    if self._check_collision((bx, by), BULLET_SIZE, enemy, ENEMY_SIZE):
                        hit_enemies.append(enemy)
                        reward += REWARD_HIT_ENEMY
                        self.kill_count += 1
                self.enemies = [e for e in self.enemies if e not in hit_enemies]
        self.bullets = new_bullets

        # 敌机逻辑
        new_enemies = []
        for enemy in self.enemies:
            ex, ey = enemy
            ey += ENEMY_SPEED
            if ey < WINDOW_HEIGHT:
                new_enemies.append((ex, ey))
                if self._check_collision((self.player_x, self.player_y), PLAYER_SIZE, (ex, ey), ENEMY_SIZE):
                    reward += REWARD_COLLIDE
                    self.player_live = False
                    done = True
        self.enemies = new_enemies

        if len(self.enemies) < MAX_ENEMIES:
            self._spawn_enemies(MAX_ENEMIES - len(self.enemies))

        if self.player_live:
            reward += REWARD_SURVIVE
        if not self.enemies and self.player_live:
            reward += REWARD_KILL_ALL
            done = True

        self.total_reward += reward
        new_state = self._get_state()
        return new_state, reward, done


# -------------------------- Q-Learning 智能体（带保存/加载） --------------------------
class QLearningAgent:
    def __init__(self, load_from_file=None):
        if load_from_file:
            # 加载预训练Q表
            try:
                with open(load_from_file, "rb") as f:
                    self.q_table = pickle.load(f)
                print(f"✅ 已加载预训练Q表: {load_from_file}")
                self.epsilon = EPSILON_END  # 加载后直接用最小探索率
            except FileNotFoundError:
                print(f"⚠️ 未找到Q表文件 {load_from_file}，从头训练")
                self.q_table = {}
                self.epsilon = EPSILON_START
        else:
            self.q_table = {}
            self.epsilon = EPSILON_START

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(ACTIONS)
        return self.q_table[state][action]

    def choose_action(self, state):
        self.get_q_value(state, 0)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(ACTIONS)
        else:
            q_vals = self.q_table[state]
            max_q = max(q_vals)
            best_acts = [i for i, v in enumerate(q_vals) if v == max_q]
            return random.choice(best_acts)

    def learn(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        next_qs = [self.get_q_value(next_state, a) for a in ACTIONS]
        max_next_q = max(next_qs)
        new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def save_q_table(self, filename="q_table_final.pkl"):
        """保存Q表到文件"""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"✅ Q表已保存到: {filename} (共{len(self.q_table)}个状态)")


# -------------------------- 可视化图表生成函数 --------------------------
def generate_training_plots(episodes_list, rewards_list, kills_list, epsilon_list, q_size_list):
    """
    生成训练效果可视化图表
    :param episodes_list: 轮次数组
    :param rewards_list: 每轮奖励数组
    :param kills_list: 每轮击毁数数组
    :param epsilon_list: 每轮探索率数组
    :param q_size_list: 每轮Q表大小数组
    """
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建2x2子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Q-Learning 训练效果 ({len(episodes_list)}轮)', fontsize=16)

    # 1. 每轮奖励曲线（加滑动平均）
    ax1.plot(episodes_list, rewards_list, color='#FF4B5C', alpha=0.5, label='单轮奖励')
    # 滑动平均（平滑曲线）
    window_size = max(1, len(episodes_list) // 20)
    if len(rewards_list) >= window_size:
        reward_smooth = np.convolve(rewards_list, np.ones(window_size) / window_size, mode='valid')
        ax1.plot(episodes_list[window_size - 1:], reward_smooth, color='#FF4B5C', linewidth=2,
                 label=f'{window_size}轮平均')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('累计奖励')
    ax1.set_title('奖励变化趋势')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. 击毁敌机数
    ax2.plot(episodes_list, kills_list, color='#00BFFF', alpha=0.7, marker='.', markersize=2)
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('击毁敌机数')
    ax2.set_title('击毁数量变化')
    ax2.grid(True, alpha=0.3)

    # 3. 探索率衰减
    ax3.plot(episodes_list, epsilon_list, color='#FFD700', linewidth=2)
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('探索率 (Epsilon)')
    ax3.set_title('探索率衰减曲线')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    # 4. Q表大小增长
    ax4.plot(episodes_list, q_size_list, color='#32CD32', linewidth=2)
    ax4.set_xlabel('训练轮次')
    ax4.set_ylabel('Q表状态数')
    ax4.set_title('Q表大小变化')
    ax4.grid(True, alpha=0.3)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()  # 弹出图表窗口
    print("✅ 训练图表已保存到: training_results.png")


# -------------------------- 主训练函数 --------------------------
def train_background(episodes=1000, save_interval=200):
    """
    后台训练主函数
    :param episodes: 训练轮数
    :param save_interval: 每多少轮保存一次Q表
    """
    # 初始化环境和智能体
    env = AirShooterEnv()
    agent = QLearningAgent()  # 从头训练 | 续训：QLearningAgent(load_from_file="q_table_final.pkl")

    # 记录训练数据（用于绘图）
    episodes_list = []
    rewards_list = []
    kills_list = []
    epsilon_list = []
    q_size_list = []

    start_time = time.time()

    # 训练日志头部
    print("=" * 80)
    print(f"开始后台训练 | 总轮数: {episodes} | 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"{'轮次':<6} {'奖励':<8} {'击毁':<6} {'探索率':<8} {'Q表大小':<8} {'耗时(s)':<8}")
    print("-" * 80)

    # 核心训练循环
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        # 单轮训练
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            agent.decay_epsilon()
            state = next_state

        # 记录本轮数据
        episodes_list.append(episode)
        rewards_list.append(env.total_reward)
        kills_list.append(env.kill_count)
        epsilon_list.append(agent.epsilon)
        q_size_list.append(len(agent.q_table))

        # 每10轮输出进度
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(rewards_list[-10:])
            print(
                f"{episode:<6} {avg_reward:<8.1f} {np.mean(kills_list[-10:]):<6.1f} {agent.epsilon:<8.3f} {len(agent.q_table):<8} {elapsed_time:<8.1f}")

        # 定期保存Q表
        if episode % save_interval == 0:
            agent.save_q_table(f"q_table_checkpoint_{episode}.pkl")

    # 训练完成
    total_time = time.time() - start_time
    print("-" * 80)
    print(f"训练完成 | 总耗时: {total_time:.1f}秒 | 平均每轮: {total_time / episodes:.3f}秒")
    print(f"最终Q表大小: {len(agent.q_table)} | 最终探索率: {agent.epsilon:.4f}")
    print(f"最后100轮平均奖励: {np.mean(rewards_list[-100:]):.1f}")
    print("=" * 80)

    # 保存最终Q表
    agent.save_q_table()

    # 生成可视化图表
    print("\n📊 正在生成训练效果图表...")
    generate_training_plots(episodes_list, rewards_list, kills_list, epsilon_list, q_size_list)

    return agent  # 返回训练好的智能体


# -------------------------- 使用训练好的Q表（测试函数） --------------------------
def test_trained_agent(q_table_path="q_table_final.pkl", test_episodes=10):
    """
    测试训练好的AI模型
    :param q_table_path: Q表文件路径
    :param test_episodes: 测试轮数
    """
    print("\n🎮 开始测试训练好的AI模型...")
    print("=" * 60)

    env = AirShooterEnv()
    # 加载训练好的Q表，探索率设为0（完全使用最优策略）
    agent = QLearningAgent(load_from_file=q_table_path)
    agent.epsilon = 0.0  # 关闭探索，只使用学到的策略

    total_rewards = []
    total_kills = []

    for episode in range(1, test_episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state

        total_rewards.append(env.total_reward)
        total_kills.append(env.kill_count)
        print(f"测试轮 {episode:2d} | 奖励: {env.total_reward:4.0f} | 击毁: {env.kill_count:2d}")

    print("=" * 60)
    print(f"测试总结 | 平均奖励: {np.mean(total_rewards):.1f} | 平均击毁: {np.mean(total_kills):.1f}")


if __name__ == "__main__":
    # 1. 执行后台训练（1000轮，可调整）
    trained_agent = train_background(episodes=1000, save_interval=200)

    # 2. 测试训练好的模型
    test_trained_agent(q_table_path="q_table_final.pkl", test_episodes=10)