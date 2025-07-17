import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def schwefel_function(x, y ):
    """ Schwefel 函數公式，適用於 x 和 y"""
    
    return 418.9829 *  2 - (x * np.sin(np.sqrt(abs(x))) + y * np.sin(np.sqrt(abs(y))))

class Particle:
    def __init__(self, dimensions):
        """
        self.position：粒子的位置，初始化在 -500 到 500 之間。
        self.velocity：粒子的速度，初始化在 -1 到 1 之間。
        self.best_position：個體最佳位置，初始值為當前位置。
        self.best_fitness：個體最佳適應度，初始值為當前位置的適應度值。
        """
        self.position = np.random.uniform(-500, 500, dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = self.evaluate(self.position)

    def evaluate(self, position):
        # Schwefel 函數適應度計算 (只取前兩個維度來計算適應度)
        return schwefel_function(position[0], position[1])

    def update_personal_best(self):
        fitness = self.evaluate(self.position)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()

class PSO:
    def __init__(self, population_size=100, dimensions = 30, w=0.5, c1=1.5, c2=1.5, max_iter=200000):
        self.population_size = population_size
        self.dimensions = dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.particles = [Particle(dimensions) for _ in range(population_size)]
        self.global_best_position = self.particles[0].position.copy()
        self.global_best_fitness = float('inf')

    def optimize(self):
        fitness_history = []
        best_fitness_history = []
        no_improvement_count = 0
        max_no_improvement = 50  # 可自定義無改善次數上限

        for i in range(self.max_iter):
            for particle in self.particles:
                # 更新個體的最佳位置
                particle.update_personal_best()

                # 更新全局最佳位置
                if particle.best_fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_position = particle.best_position.copy()

            # 更新速度和位置
            for particle in self.particles:
                inertia = self.w * particle.velocity
                cognitive = self.c1 * np.random.rand(self.dimensions) * (particle.best_position - particle.position)
                social = self.c2 * np.random.rand(self.dimensions) * (self.global_best_position - particle.position)

                particle.velocity = inertia + cognitive + social
                particle.position += particle.velocity

                # 限制粒子位置範圍在 [-500, 500] 之間
                particle.position = np.clip(particle.position, -500, 500)

            fitness_history.append(self.global_best_fitness)
            best_fitness_history.append(self.global_best_fitness)
            print(f"世代 {i}, 全局最佳適應度: {self.global_best_fitness}")

            # 如果最佳適應度沒有改善
            if len(best_fitness_history) > 1 and abs(best_fitness_history[-1] - best_fitness_history[-2]) < 1e-6:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            if no_improvement_count >= max_no_improvement:
                print("無改善達到最大次數，提前退出!")
                break
                
        return self.global_best_position, fitness_history

    def plot_3d_surface(self):
        x = np.linspace(-500, 500, 200)
        y = np.linspace(-500, 500, 200)
        x, y = np.meshgrid(x, y)
        z = schwefel_function(x, y)  # Schwefel 函數的 z 值計算

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='k', alpha=0.7)

        # 繪製粒子群及最佳位置
        for particle in self.particles[0:2]:
            # 只使用前兩個維度，計算 Schwefel 函數的表面 z 值
            x_value = particle.position[0]
            y_value = particle.position[1]
            z_value = schwefel_function(x_value, y_value)  # Schwefel 表面上的 z 值

            ax.scatter(x_value, y_value, z_value, c='r', s=100)  # 粒子位於 Schwefel 函數表面

        # 繪製全局最佳點
        ax.scatter(self.global_best_position[0], self.global_best_position[1], self.global_best_fitness, c='b', s=150, label='best')

        # 標記每個粒子點的適應度值
        for particle in self.particles:
            x_value = particle.position[0]
            y_value = particle.position[1]
            z_value = schwefel_function(x_value, y_value)  # Schwefel 表面上的 z 值
            # ax.text(x_value, y_value, z_value, f'{z_value:.2f}', color='black')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Fitness')
        ax.set_title('Schwefel')
        ax.set_zlim(-2000, 2000)
        plt.legend()
        plt.show()


# 主程式
if __name__ == "__main__":
    pso = PSO(population_size=100, dimensions=30, max_iter=200000)
    best_position, fitness_history = pso.optimize()
    print(f"最終最佳位置: {best_position}")
    print(f"全局最佳適應度: {pso.global_best_fitness}")
    pso.plot_3d_surface()
