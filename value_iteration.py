"""
贝尔曼方程的形式有两种,值迭代采取的是状态-动作值函数的形式
这里仍然是基于模型的,确定性策略,我们认为状态s采取动作a后到达状态s'的概率为1
"""


import numpy as np
from env import GridWorld

# 值迭代
def value_iteration(grid_world = GridWorld(5,5),gamma=0.9,theta=1e-6):
    policy = {state:{action:1/len(grid_world.actions) for action in grid_world.actions} for state in grid_world.states}  
    V = np.zeros((grid_world.height,grid_world.width))

    # 计算出该状态的所有可能的v，并取最大的v，而非再进行期望的计算
    while True:
        delta = 0
        for state in grid_world.states:
            if state != grid_world.goal_state:
                v_a = []
                for action in grid_world.actions:
                    prob,next_state,reward = grid_world.step(state,action)
                    v = prob*(reward+gamma*V[next_state])
                    v_a.append(v)
                delta = max(delta,np.abs(max(v_a)-V[state]))
                V[state] = max(v_a)
        if delta<theta:
            break
    # return V


    # 此处步骤同策略迭代的第二步策略改进,选择最大状态-动作值对应的动作
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

if __name__ == "__main__":

    width = 10
    height = 10
    grid_world = GridWorld(width, height)
    grid_world.plot_grid()

    print(grid_world.r_grid)
    policy,V = value_iteration(grid_world)

    mapping = {(-1,0):'↑',(1,0):'↓',(0,-1):'←',(0,1):'→'}
    best_actions = np.empty((height, width),dtype=object)

    for state,action in policy.items():
        best_action = max(action,key=action.get)
        best_actions[state]=mapping[best_action]
        if grid_world.r_grid[state]==-1:
            best_actions[state] = '□'
    
    print(best_actions)

    grid_world.plot_trajectory((0,0),policy,invisible=False)
    