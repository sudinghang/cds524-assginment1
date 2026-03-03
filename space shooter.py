import pygame
import numpy as np
import random
import pickle  # 新增：用于加载Q表

# -------------------------- 初始化Pygame（基础准备） --------------------------
pygame.init()
pygame.font.init()

# -------------------------- 游戏核心参数配置 --------------------------
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("迷你太空射击游戏 - Q-Learning版")
CLOCK = pygame.time.Clock()
FPS = 20

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 191, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)

# 尺寸/速度参数
PLAYER_SIZE = 30
PLAYER_SPEED = 5
ENEMY_SIZE = 25
ENEMY_SPEED = 2
BULLET_SIZE = 5
BULLET_SPEED = 15

# -------------------------- Q-Learning 强化学习参数 --------------------------
ACTIONS = [0, 1, 2, 3, 4]
# WASD方向映射
ACTION_DELTA = [
    (0, -PLAYER_SPEED),  # 0=上(W)
    (0, PLAYER_SPEED),  # 1=下(S)
    (-PLAYER_SPEED, 0),  # 2=左(A)
    (PLAYER_SPEED, 0),  # 3=右(D)
    (0, 0)  # 4=射击(空格)
]

# Q-Learning核心参数（和训练代码一致）
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
ALPHA = 0.2
GAMMA = 0.9

# 奖励参数（核心）
REWARD_HIT_ENEMY = 150
REWARD_COLLIDE = -300
REWARD_SHOOT = -1
REWARD_SURVIVE = 5
REWARD_OUT_BOUND = -20
REWARD_KILL_ALL = 1000

MAX_ENEMIES = 3
MAX_BULLETS = 3


# -------------------------- 加载战机图片（新增） --------------------------
def load_images():
    """加载战机/敌机/子弹图片，加载失败则用图形替代"""
    player_img = None
    enemy_img = None
    bullet_img = None

    try:
        # 加载图片并缩放至指定尺寸
        player_img = pygame.image.load("player.png").convert_alpha()
        player_img = pygame.transform.scale(player_img, (PLAYER_SIZE, PLAYER_SIZE))

        enemy_img = pygame.image.load("enemy.png").convert_alpha()
        enemy_img = pygame.transform.scale(enemy_img, (ENEMY_SIZE, ENEMY_SIZE))

        bullet_img = pygame.image.load("bullet.png").convert_alpha()
        bullet_img = pygame.transform.scale(bullet_img, (BULLET_SIZE, BULLET_SIZE))
        print("✅ 图片加载成功！")
    except:
        print("⚠️ 未找到图片文件，使用图形替代")

    return player_img, enemy_img, bullet_img


# 全局图片变量
PLAYER_IMG, ENEMY_IMG, BULLET_IMG = load_images()


# -------------------------- 游戏环境类（含完整奖励机制） --------------------------
class AirShooterEnv:
    def __init__(self):
        # 初始化飞船位置
        self.player_x = WINDOW_WIDTH // 2 - PLAYER_SIZE // 2
        self.player_y = WINDOW_HEIGHT - PLAYER_SIZE - 10
        self.player_live = True

        # 初始化敌机和子弹
        self.enemies = []
        self._spawn_enemies(MAX_ENEMIES)
        self.bullets = []

        # 游戏统计
        self.kill_count = 0
        self.total_reward = 0

    def _spawn_enemies(self, num):
        """生成指定数量的敌机，避免重叠"""
        for _ in range(num):
            while True:
                enemy_x = random.randint(ENEMY_SIZE, WINDOW_WIDTH - ENEMY_SIZE)
                enemy_y = random.randint(-100, -ENEMY_SIZE)
                enemy = (enemy_x, enemy_y)
                # 检查是否与已有敌机重叠
                if not any([abs(enemy[0] - e[0]) < ENEMY_SIZE and abs(enemy[1] - e[1]) < ENEMY_SIZE for e in
                            self.enemies]):
                    self.enemies.append(enemy)
                    break

    def _is_out_bound(self, x, y, size):
        """判断物体是否越界"""
        return x < 0 or x > WINDOW_WIDTH - size or y < 0 or y > WINDOW_HEIGHT - size

    def _check_collision(self, obj1, obj1_size, obj2, obj2_size):
        """检测两个矩形物体是否碰撞"""
        x1, y1 = obj1
        x2, y2 = obj2
        return (x1 < x2 + obj2_size and x1 + obj1_size > x2 and
                y1 < y2 + obj2_size and y1 + obj1_size > y2)

    def _calculate_reward(self, action):
        """
        集中封装奖励计算逻辑（核心奖励机制）
        :param action: 当前执行的动作（-1表示无主动动作）
        :return: 综合奖励值
        """
        reward = 0

        # 1. 生存奖励：每帧存活即获得基础奖励
        if self.player_live:
            reward += REWARD_SURVIVE

        # 2. 越界惩罚：防止飞船停在边缘避战
        if self._is_out_bound(self.player_x, self.player_y, PLAYER_SIZE):
            reward += REWARD_OUT_BOUND

        # 3. 射击惩罚：避免无意义乱射，引导精准攻击
        if action == 4 and len(self.bullets) < MAX_BULLETS:
            reward += REWARD_SHOOT

        # 4. 碰撞惩罚：强烈抑制与敌机碰撞的行为
        player_pos = (self.player_x, self.player_y)
        for enemy in self.enemies:
            if self._check_collision(player_pos, PLAYER_SIZE, enemy, ENEMY_SIZE):
                reward += REWARD_COLLIDE
                break  # 只惩罚一次碰撞

        # 5. 击中敌机奖励：引导主动攻击核心目标
        hit_count = 0
        for bullet in self.bullets:
            for enemy in self.enemies:
                if self._check_collision(bullet, BULLET_SIZE, enemy, ENEMY_SIZE):
                    hit_count += 1
        reward += hit_count * REWARD_HIT_ENEMY

        # 6. 通关奖励：击落所有敌机的终极奖励
        if not self.enemies and self.player_live:
            reward += REWARD_KILL_ALL

        return reward

    def reset(self):
        """重置游戏状态（新增：返回真实状态，用于Q-Learning）"""
        self.__init__()
        # 计算当前状态（和训练代码的_get_state逻辑完全一致）
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
        """执行动作并更新游戏状态（返回真实状态）"""
        reward = 0
        done = False

        # 处理飞船移动/射击动作
        dx, dy = ACTION_DELTA[action]
        if action in [0, 1, 2, 3]:
            new_x = self.player_x + dx
            new_y = self.player_y + dy
            if not self._is_out_bound(new_x, new_y, PLAYER_SIZE):
                self.player_x = new_x
                self.player_y = new_y
        elif action == 4:
            if len(self.bullets) < MAX_BULLETS:
                bullet_x = self.player_x + PLAYER_SIZE // 2 - BULLET_SIZE // 2
                bullet_y = self.player_y - BULLET_SIZE
                self.bullets.append((bullet_x, bullet_y))

        # 子弹逻辑：移动+击中检测
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
                        self.kill_count += 1
                self.enemies = [e for e in self.enemies if e not in hit_enemies]
        self.bullets = new_bullets

        # 敌机逻辑：移动+碰撞检测
        new_enemies = []
        for enemy in self.enemies:
            ex, ey = enemy
            ey += ENEMY_SPEED
            if ey < WINDOW_HEIGHT:
                new_enemies.append((ex, ey))
                if self._check_collision((self.player_x, self.player_y), PLAYER_SIZE, (ex, ey), ENEMY_SIZE):
                    self.player_live = False
                    done = True
        self.enemies = new_enemies

        # 补充生成敌机，保持数量
        if len(self.enemies) < MAX_ENEMIES:
            self._spawn_enemies(MAX_ENEMIES - len(self.enemies))

        # 计算奖励（核心：调用封装的奖励函数）
        reward = self._calculate_reward(action)

        # 游戏结束判断
        if not self.player_live:
            done = True
        if not self.enemies and self.player_live:
            done = True

        self.total_reward += reward

        # 计算新状态（和训练代码一致）
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
        next_state = (player_x_seg, player_y_seg, enemy_x_seg, enemy_y_seg, bullet_exist)

        return next_state, reward, done

    def update_game_state(self):
        """无主动动作时更新游戏状态（手动模式）"""
        reward = 0
        done = False

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
                    self.player_live = False
                    done = True
        self.enemies = new_enemies

        # 补充生成敌机
        if len(self.enemies) < MAX_ENEMIES:
            self._spawn_enemies(MAX_ENEMIES - len(self.enemies))

        # 计算奖励（无主动动作，传-1）
        reward = self._calculate_reward(action=-1)

        # 游戏结束判断
        if not self.player_live:
            done = True
        if not self.enemies and self.player_live:
            done = True

        self.total_reward += reward
        return (0, 0, 0, 0, 0), reward, done


# -------------------------- Q-Learning 智能体（完整版，支持加载Q表） --------------------------
class QLearningAgent:
    def __init__(self, load_from_file=None):
        if load_from_file:
            # 加载预训练Q表
            try:
                with open(load_from_file, "rb") as f:
                    self.q_table = pickle.load(f)
                print(f"✅ 已加载预训练Q表: {load_from_file}")
                self.epsilon = EPSILON_END  # 加载后几乎不探索
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


# -------------------------- UI 绘制函数（新增：绘制战机图片） --------------------------
def draw_menu():
    """绘制开始菜单"""
    SCREEN.fill(WHITE)
    try:
        title_font = pygame.font.SysFont("arial", 48)
        text_font = pygame.font.SysFont("SimHei", 20)
    except:
        title_font = pygame.font.SysFont(None, 48)
        text_font = pygame.font.SysFont(None, 20)

    title = title_font.render("Space Shooter", True, BLACK)
    desc1 = text_font.render("手动操控飞船击落敌机", True, GREEN)
    desc2 = text_font.render("按 Y 开始", True, BLACK)
    quit_t = text_font.render("ESC 退出", True, BLACK)

    SCREEN.blit(title, title.get_rect(center=(WINDOW_WIDTH // 2, 100)))
    SCREEN.blit(desc1, desc1.get_rect(center=(WINDOW_WIDTH // 2, 200)))
    SCREEN.blit(desc2, desc2.get_rect(center=(WINDOW_WIDTH // 2, 250)))
    SCREEN.blit(quit_t, quit_t.get_rect(center=(WINDOW_WIDTH // 2, 300)))
    pygame.display.flip()


def draw_game(env, manual_mode):
    """绘制游戏主界面（替换为战机图片）"""
    SCREEN.fill(BLACK)
    # 绘制玩家战机
    if env.player_live:
        if PLAYER_IMG:
            SCREEN.blit(PLAYER_IMG, (env.player_x, env.player_y))
        else:
            # 备用：蓝色三角形战机（替代方块）
            pygame.draw.polygon(SCREEN, BLUE, [
                (env.player_x + PLAYER_SIZE // 2, env.player_y),
                (env.player_x, env.player_y + PLAYER_SIZE),
                (env.player_x + PLAYER_SIZE, env.player_y + PLAYER_SIZE)
            ])

    # 绘制敌机
    for e in env.enemies:
        if ENEMY_IMG:
            SCREEN.blit(ENEMY_IMG, (e[0], e[1]))
        else:
            # 备用：红色菱形敌机（替代方块）
            center_x = e[0] + ENEMY_SIZE // 2
            center_y = e[1] + ENEMY_SIZE // 2
            pygame.draw.polygon(SCREEN, RED, [
                (center_x, center_y - ENEMY_SIZE // 2),
                (center_x + ENEMY_SIZE // 2, center_y),
                (center_x, center_y + ENEMY_SIZE // 2),
                (center_x - ENEMY_SIZE // 2, center_y)
            ])

    # 绘制子弹
    for b in env.bullets:
        if BULLET_IMG:
            SCREEN.blit(BULLET_IMG, (b[0], b[1]))
        else:
            # 备用：黄色圆形子弹（替代方块）
            pygame.draw.circle(SCREEN, YELLOW,
                               (b[0] + BULLET_SIZE // 2, b[1] + BULLET_SIZE // 2),
                               BULLET_SIZE)

    # 绘制统计信息
    try:
        font = pygame.font.SysFont("SimHei", 20)
    except:
        font = pygame.font.SysFont(None, 20)

    r_t = font.render(f"累计奖励: {env.total_reward}", True, GREEN)
    k_t = font.render(f"击毁敌机: {env.kill_count}", True, WHITE)
    mode = "手动模式" if manual_mode else "自动模式"
    m_t = font.render(f"当前模式: {mode}", True, YELLOW)

    SCREEN.blit(r_t, (10, 10))
    SCREEN.blit(k_t, (10, 40))
    SCREEN.blit(m_t, (10, 70))
    pygame.display.flip()


def draw_game_over(env, episode):
    """绘制游戏结束界面"""
    SCREEN.fill(WHITE)
    try:
        t_font = pygame.font.SysFont("arial", 48)
        font = pygame.font.SysFont("SimHei", 20)
    except:
        t_font = pygame.font.SysFont(None, 48)
        font = pygame.font.SysFont(None, 20)

    go = t_font.render("Game Over", True, BLACK)
    ep = font.render(f"游戏轮次: {episode}", True, GREEN)
    rw = font.render(f"累计奖励: {env.total_reward}", True, BLACK)
    kl = font.render(f"击毁敌机: {env.kill_count}", True, BLACK)
    r_t = font.render("按 R 重开", True, BLACK)
    esc_t = font.render("ESC 返回菜单", True, BLACK)

    SCREEN.blit(go, go.get_rect(center=(WINDOW_WIDTH // 2, 150)))
    SCREEN.blit(ep, ep.get_rect(center=(WINDOW_WIDTH // 2, 220)))
    SCREEN.blit(rw, rw.get_rect(center=(WINDOW_WIDTH // 2, 260)))
    SCREEN.blit(kl, kl.get_rect(center=(WINDOW_WIDTH // 2, 300)))
    SCREEN.blit(r_t, r_t.get_rect(center=(WINDOW_WIDTH // 2, 350)))
    SCREEN.blit(esc_t, esc_t.get_rect(center=(WINDOW_WIDTH // 2, 390)))
    pygame.display.flip()


# -------------------------- 主函数（支持加载Q表） --------------------------
def main():
    # 显示开始菜单
    in_menu = True
    while in_menu:
        CLOCK.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    in_menu = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        draw_menu()

    # 初始化游戏环境和智能体（关键：加载训练好的Q表）
    env = AirShooterEnv()
    # 替换成你的Q表文件名，比如 q_table_checkpoint_800.pkl
    agent = QLearningAgent(load_from_file="q_table_checkpoint_1000.pkl")
    state = env.reset()

    # 游戏主循环
    running = True
    episode = 0
    manual_mode = True  # 默认手动模式
    done = False

    while running:
        CLOCK.tick(FPS)

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # 重开游戏
                    state = env.reset()
                    done = False
                elif event.key == pygame.K_ESCAPE:  # 返回菜单
                    return main()
                elif event.key == pygame.K_m:  # 切换手动/自动模式
                    manual_mode = not manual_mode

        # 处理游戏逻辑
        action = None
        if manual_mode:
            # 手动模式：WASD移动，空格射击
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                action = 0
            elif keys[pygame.K_s]:
                action = 1
            elif keys[pygame.K_a]:
                action = 2
            elif keys[pygame.K_d]:
                action = 3
            elif keys[pygame.K_SPACE]:
                action = 4
        else:
            # 自动模式：AI使用Q表选择动作
            action = agent.choose_action(state)
            # AI学习（可选：注释掉则AI不继续学习，只使用已有Q表）
            # next_state, reward, done = env.step(action)
            # agent.learn(state, action, reward, next_state)
            # agent.decay_epsilon()
            # state = next_state

        # 更新游戏状态
        if action is not None:
            next_state, reward, done = env.step(action)
            # 自动模式下让AI学习（可选）
            if not manual_mode:
                agent.learn(state, action, reward, next_state)
                agent.decay_epsilon()
            state = next_state
        else:
            next_state, reward, done = env.update_game_state()
            state = next_state

        # 绘制游戏界面
        draw_game(env, manual_mode)

        # 游戏结束处理
        if done:
            episode += 1
            in_go = True
            while in_go:
                CLOCK.tick(FPS)
                draw_game_over(env, episode)
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        running = False
                        in_go = False
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_r:  # 重开
                            state = env.reset()
                            done = False
                            in_go = False
                        elif e.key == pygame.K_ESCAPE:  # 返回菜单
                            return main()

    # 退出游戏
    pygame.quit()
    pygame.font.quit()


if __name__ == "__main__":
    main()