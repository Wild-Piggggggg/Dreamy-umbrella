import numpy as np
import random
from env import GridWorld


"""
在很多场景下,马尔科夫决策过程(MDP)的状态转移概率和奖励函数r(s,a,s')是未知的,
这样就需要先采样,再根据这些样本来求解最优策略.
"""


# 定义随机策略
def random_policy(grid_world,state): # 这里其实不传state也行...
    return random.choice(grid_world.actions)


# 定义ε-greedy策略
def epsilon_greedy_policy(grid_world,state,Q,epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(grid_world.actions)
    else:
        return max(Q[state],key=Q[state].get)
    

def MC(grid_world,num_episodes=10000,gamma=0.9,epsilon=0.1):  # MC方法对数据量的要求很大，需要足够大的迭代次数，否则效果会比较差
    # 初始化状态值函数
    V = np.zeros((grid_world.height,grid_world.width))

    # 初始化状态-动作值函数和奖励计数器
    Q = {}
    G = {}
    for state in grid_world.states:
        Q[state] = {action:0 for action in grid_world.actions}
        G[state] = {action:[] for action in grid_world.actions}
    
    # 开始蒙特卡洛过程
    for episode in range(1,num_episodes+1):
        # 随机选择一个起始状态
        start_state = random.choice(grid_world.states)

        # 由随机策略选择一个动作
        action = epsilon_greedy_policy(grid_world,start_state,Q)

        # 保存轨迹和动作
        trajectory = []
        # cumulative_reward = 0

        # 在当前随机选择的起始状态下执行动作,收集一条完整的轨迹
        state = start_state
        while True:
            # 处理边界情况，如果next_state和state相同，就说明到边界了
            prob,next_state,reward = grid_world.step(state,action)

            if state==next_state:
                break

            trajectory.append((state,action,reward))

            if next_state == grid_world.goal_state:
                break
            action = epsilon_greedy_policy(grid_world,next_state,Q)
            state = next_state
        
        # 计算每个状态-动作对的回报
        # 由于可能存在多次访问同一个状态-动作对的情况，
        # 因此我们将每次获得的累积奖励都添加到对应的 G[state][action] 列表中，以便后续计算平均值

        cumulative_reward = 0
        for state,action,reward in trajectory[::-1]:
            cumulative_reward = gamma*cumulative_reward + reward
            G[state][action].append(cumulative_reward)
        
        # 更新状态值函数和动作值函数
        for state in G:
            for action in G[state]:
                if G[state][action]:
                    Q[state][action] = np.mean(G[state][action])
                    V[state] = max(Q[state].values())

        if episode%1000==0:
            print(V)

    policy = {state:{action:1/len(grid_world.actions) for action in grid_world.actions} for state in grid_world.states}
    for state in grid_world.states:
        # 不处理终止状态相当于到达终止状态(
        if state != grid_world.goal_state:
            q_values = {action:0 for action in grid_world.actions}
            for action in grid_world.actions:
                prob,next_state,reward = grid_world.step(state,action)
                q_values[action] = prob*(reward+gamma*V[next_state])

            best_action = max(q_values,key=q_values.get)

            # 选择策略中具有最大状态-动作值的动作对应的概率设置为1,其他赋为0
            for a in policy[state].keys():
                if a==best_action:
                    policy[state][a] = 1
                else:
                    policy[state][a] = 0
    
    return policy,V

# 示例运行
if __name__ == "__main__":

    width = 6
    height = 5
    grid_world = GridWorld(width, height)
    # grid_world.plot_grid()

    print(grid_world.r_grid)
    policy,V = MC(grid_world)

    mapping = {(-1,0):'↑',(1,0):'↓',(0,-1):'←',(0,1):'→'}
    best_actions = np.empty((height, width),dtype=object)

    for state,action in policy.items():
        best_action = max(action,key=action.get)
        best_actions[state]=mapping[best_action]
        if grid_world.r_grid[state]==-1:
            best_actions[state] = '□'
    
    print(best_actions)

    grid_world.plot_trajectory((0,0),policy,invisible=False)