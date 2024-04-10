"""格子世界代码"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time,copy


class GridWorld:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # 定义奖励函数，只要进到这个格子，就获得对应的奖励
        self.r_grid = np.zeros((height, width))
        self.start_state = (0, 0)
        self.goal_state = (height - 1, width - 1)
        self.obstacles = self.obstacles_generate()  # 定义障碍物位置,随机取样,大概占据一半吧
        
        for obstacle in self.obstacles:
            self.r_grid[obstacle] = -1
        
        self.r_grid[self.start_state] = 0
        self.r_grid[self.goal_state] = 1

        self.actions = [(-1,0),(1,0),(0,-1),(0,1)]  # 上下左右
        self.grid = copy.deepcopy(self.r_grid)  # 这里浅拷贝
        # self.grid[self.start_state] = 1  # 为了绘图起点也显示出来，好看
        self.states = [(i,j) for i in range(height) for j in range(width)]


    # 生成障碍格子
    def obstacles_generate(self):
        L_r = range(self.height)
        L_c = range(self.width)
        ob = []
        ob_nums = int(self.width*self.height/3)
        for _ in range(ob_nums):
            a = random.choice(L_r)
            b = random.choice(L_c)
            ob.append((a,b))

        return list(set(ob))
    
    
    # 重置状态
    def reset(self):
        return self.start_state
    

    # 根据动作移动一步，返回新的状态和奖励及结束状态
    # 碰壁与碰到障碍的奖励都为-1,如果碰壁就回到原位
    def step(self, state, action):

        next_state = (state[0] + action[0], state[1] + action[1])
        if -1<next_state[0]<self.height and -1<next_state[1]<self.width:
            reward = self.r_grid[next_state]
            pass
        else:
            next_state = state
            reward = 0
        
        return [1.0,next_state,reward]


    # 绘制网格
    def plot_grid(self,invisible=True):
        """"
        invisible:是否将每一步可视化,默认不可视化"""
        fig,ax = plt.subplots()
        colors = ['green','orange', 'blue']  # 分别表示普通格子、起始格子和终点格子的颜色
        cmap = mcolors.ListedColormap(colors)

        plt.imshow(self.grid, cmap=cmap, interpolation='nearest')
        plt.xticks(np.arange(0, self.width, 1))
        plt.yticks(np.arange(0, self.height, 1))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.show(block=False)
        plt.pause(1)
        plt.close()


    # 从任意状态出发，找到它到达终点的轨迹
    def find_trajectory(self,start_state,policy):
        
        trajectory = [start_state]
        state = start_state
        for _ in range(self.width*self.height):  # 最多把所有格子都遍历一遍
            action = max(policy[state],key=policy[state].get)
            state = (state[0]+action[0],state[1]+action[1])

            if state == self.goal_state :
                trajectory.append(state)
                break

            if -1<state[0]<self.height and -1<state[1]<self.width:
                trajectory.append(state)
            else:
                break
        
        return trajectory
    

    # 绘制运动轨迹
    def plot_trajectory(self,start_state,policy,invisible=True):
        """"
        invisible:是否将每一步可视化,默认不可视化"""
        fig,ax = plt.subplots()
        colors = ['green','orange', 'blue']  # 分别表示普通格子、起始格子和终点格子的颜色
        cmap = mcolors.ListedColormap(colors)

        self.grid[start_state] = 1  # 这里将起点的值设为1,是为了在最终的可视化阶段能看到起点是蓝色的,但是它在轨迹中的值还是0

        plt.imshow(self.grid, cmap=cmap, interpolation='nearest')
        plt.xticks(np.arange(0, self.width, 1))
        plt.yticks(np.arange(0, self.height, 1))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

        self.trajectory = self.find_trajectory(start_state,policy)

        # 绘制代理程序的运动轨迹,注意行列和镜像关系
        trajectory_x = [pos[1] for pos in self.trajectory]
        trajectory_y = [pos[0] for pos in self.trajectory]
        
        if invisible:
            plt.plot(trajectory_x, trajectory_y, color='red', linewidth=2)

        else:
            for i in range(len(trajectory_x)):
                # ax.clear()
                # ax.imshow(self.grid, cmap=cmap, interpolation='nearest')
                plt.plot(trajectory_x[:i+1], trajectory_y[:i+1], color='red', linewidth=2)

                plt.title('Grid World')
                plt.draw()
                plt.pause(0.1)  # 暂停0.5秒

            # plt.title('Grid World')
        plt.show(block=False)
        plt.pause(0.3)
  
        plt.close()