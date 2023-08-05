import random
import libvecmul as vm

import matplotlib.pyplot as plt

class VecElMul:
    def __init__(self, num_elements,
            grid_dim, block_dim):

        self.num_elements = num_elements
        self.grid_dim = grid_dim
        self.block_dim = block_dim

    def naive_half(self, total_num=2000, step=20, iters=10):
        self.total_num = total_num
        self.step = step

        coordinates = []
        for x in range(1, total_num, step):
            y = vm.element_wise_mul_naive_half(self.num_elements, x, iters)
            coordinates.append((x, y))
        return coordinates

    def naive_float(self, total_num=2000, step=20, iters=10):
        self.total_num = total_num
        self.step = step

        coordinates = []
        for x in range(1, total_num, step):
            y = vm.element_wise_mul_naive_float(self.num_elements, x, iters)
            coordinates.append((x, y))
        return coordinates

    def optimized_half(self, iters=10):
        x, y = vm.element_wise_mul_optimized_half(self.num_elements, iters)
        return (x,y)

    def optimized_float(self, iters=10):
        x, y = vm.element_wise_mul_optimized_float(self.num_elements, iters)
        return (x,y)

    def draw_performance(self, points, opt, title, save_path, draw_opt=False):

        # Extract x and y coordinates from points
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        # Extract x and y coordinates from opt point
        opt_x = opt[0]
        opt_y = opt[1]

        # Create a scatter plot
        plt.scatter(x_coords, y_coords, color='blue', marker='o', label='Scattered Points')
        if draw_opt:
            plt.scatter(opt_x, opt_y, color='green', marker='o', label='Opt Point')  # Plot opt point

        # Draw lines between points
        for i in range(len(points) - 1):
            plt.plot([x_coords[i], x_coords[i + 1]], [y_coords[i], y_coords[i + 1]], color='red')

        # Find the max and min y values
        y_max = max(y_coords)

        # Find corresponding x coordinates for max and min y values
        x_max = x_coords[y_coords.index(y_max)]

        y_min = min(y_coords[int(x_max/self.step):])
        x_min = x_coords[y_coords.index(y_min)]

        # Draw vertical lines for max and min y values
        plt.axvline(x=x_max, color='green', linestyle='dashed', 
                    label=f'Max (x={x_max:.2f}, y={y_max:.2f})')
        plt.axvline(x=x_min, color='orange', linestyle='dashed', 
                    label=f'Min (x={x_min:.2f}, y={y_min:.2f})')

        # Mark the coordinates of the opt point on the image
        plt.text(opt_x, opt_y, f'({opt_x:.2f}, {opt_y:.2f})', color='green')

        # Set plot title and labels
        plt.title(title)
        plt.xlabel('grid_dim')
        plt.ylabel('efficiency')

        # Display legend
        #plt.legend()

        # Save the plot as an image file
        plt.savefig(save_path)

def test_elementwise_mul():
    num_elements = 163840*1000
    block_dim = 128
    grid_dim = 240

    vec_mul = VecElMul(num_elements, grid_dim, block_dim)

    title = 'half performance'
    coordinate_list = vec_mul.naive_half(total_num=2000)
    opt = vec_mul.optimized_half()
    vec_mul.draw_performance(coordinate_list, opt, title, './half_perf')

    title = 'float performance'
    coordinate_list = vec_mul.naive_float(total_num=2000)
    opt = vec_mul.optimized_float()
    vec_mul.draw_performance(coordinate_list, opt, title, 'float_perf')

test_elementwise_mul()



