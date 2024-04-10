"""
贝尔曼方程的形式有两种,策略迭代采取的是状态值函数的形式
由于这是基于模型的,需要给出状态转移概率p(s'|s,a)和奖励函数r(s,a,s')
这里我们知道状态转移概率为1,可以理解为状态s采用a动作一定能到s',到不了就为0,没有计算必要
奖励函数已用r_grid事先储存好了
策略的话我们这里采用随机策略,如此一来既可以采用随机生成的方式生成策略,也可以提前定义π(a|s) = 1/|A|
"""

import numpy as np
from env import GridWorld

# 策略评估:用于迭代状态值
# 根据当前的策略评估状态值
def policy_evaluation(grid_world,policy,gamma=0.9,theta=1e-6):

    V = np.zeros((grid_world.height,grid_world.width))
    while True:
        delta = 0
        for state in grid_world.states:
            # 不处理终止状态相当于到达终止状态(
            if state != grid_world.goal_state:
                v = 0
                for action,action_prob in policy[state].items():
                    prob,next_state,reward = grid_world.step(state,action)
                    v += action_prob*prob*(reward+gamma*V[next_state])  # prob定为1,确定性策略
                delta = max(delta,np.abs(v-V[state]))
                V[state] = v
        if delta<theta:
            break
    return V


# 策略改进:用于根据前一步迭代出的状态值更新策略
# 可以理解为从上一个策略得到的状态值出发，进行策略的改进，改进为确定性策略
def policy_improvement(grid_world, V, gamma=0.9):

    # 这里的policy的具体数值其实设置为多少都无所谓，只需要保证基本的结构就可以，不过这里我设置了一个随机策略
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
    
    return policy


# 策略迭代:不断重复策略评估和策略改进两个步骤
def policy_iteration(grid_world,gamma=0.9,max_iter=1000,theta=1e-6):
    policy = {state:{action:1/len(grid_world.actions) for action in grid_world.actions} for state in grid_world.states}
    for _ in range(max_iter):
        V = policy_evaluation(grid_world,policy,gamma,theta)
        # new_policy = policy_improvement(grid_world,policy,V,gamma)   # 这里要小心字典是可变对象，传入函数是会改变的，这就会导致后一步policy==new_policy的判断提前，百分百执行，所以用下面那行
        new_policy = policy_improvement(grid_world,V,gamma)  # 函数内部会定义一个新的随机策略，防止撞车,python的可变对象进入函数会改变确实是个麻烦的问题，一定要注意！！

        if policy==new_policy:
            break
        policy = new_policy

    return policy,V


# 示例运行
if __name__ == "__main__":

    width = 10
    height = 10
    grid_world = GridWorld(width, height)
    grid_world.plot_grid()

    print(grid_world.r_grid)
    policy,V = policy_iteration(grid_world)

    mapping = {(-1,0):'↑',(1,0):'↓',(0,-1):'←',(0,1):'→'}
    best_actions = np.empty((height, width),dtype=object)

    for state,action in policy.items():
        best_action = max(action,key=action.get)
        best_actions[state]=mapping[best_action]
        if grid_world.r_grid[state]==-1:
            best_actions[state] = '□'
    
    print(best_actions)

    grid_world.plot_trajectory((0,0),policy,invisible=False)