import random
import libvecmul as vm

import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw, ImageFont

class ElementWise:
    def __init__(self, num_elements,
            grid_dim, block_dim):

        self.num_elements = num_elements
        self.grid_dim = grid_dim
        self.block_dim = block_dim

    def naive_half_add(self, total_num=2000, step=20, iters=10):
        self.total_num = total_num
        self.step = step

        coordinates = []
        for x in range(1, total_num, step):
            y = vm.element_wise_add_naive_half(self.num_elements, x, iters)
            coordinates.append((x, y))
        return coordinates

    def naive_float_add(self, total_num=2000, step=20, iters=10):
        self.total_num = total_num
        self.step = step

        coordinates = []
        for x in range(1, total_num, step):
            y = vm.element_wise_add_naive_float(self.num_elements, x, iters)
            coordinates.append((x, y))
        return coordinates

    def optimized_half_add(self, iters=10):
        x, y = vm.element_wise_add_optimized_half(self.num_elements, iters)
        return (x,y)

    def optimized_float_add(self, iters=10):
        x, y = vm.element_wise_add_optimized_float(self.num_elements, iters)
        return (x,y)

    def naive_half_mul(self, total_num=2000, step=20, iters=10):
        self.total_num = total_num
        self.step = step

        coordinates = []
        for x in range(1, total_num, step):
            y = vm.element_wise_mul_naive_half(self.num_elements, x, iters)
            coordinates.append((x, y))
        return coordinates

    def naive_float_mul(self, total_num=2000, step=20, iters=10):
        self.total_num = total_num
        self.step = step

        coordinates = []
        for x in range(1, total_num, step):
            y = vm.element_wise_mul_naive_float(self.num_elements, x, iters)
            coordinates.append((x, y))
        return coordinates

    def optimized_half_mul(self, iters=10):
        x, y = vm.element_wise_mul_optimized_half(self.num_elements, iters)
        return (x,y)

    def optimized_float_mul(self, iters=10):
        x, y = vm.element_wise_mul_optimized_float(self.num_elements, iters)
        return (x,y)

    def draw_performance(self, num, points, opt, title, save_path, draw_opt=False):
        # Create a new figure
        plt.figure()

        # Plot the connected lines between points
        x_values, y_values = zip(*points)
        plt.plot(x_values, y_values, marker='o', color='blue', label='Points')

        # Find max x and y values
        max_x = max(x_values)
        max_y = max(y_values)

        # Customize the plot if needed (e.g., title, labels, grid, etc.)
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()

        # Draw the 'opt' coordinate as text
        opt_x, opt_y = opt
        plt.text(0.95, 0.05, f"Optimal: ({opt_x}, {opt_y})", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')

        # Draw max x and y values as text
        plt.text(0.95, 0.1, f"Max X: {max_x}", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')
        plt.text(0.95, 0.15, f"Max Y: {max_y}", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')

        # Draw the 'num' value as text
        plt.text(0.95, 0.2, f"Num: {num}", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')

        # Save the image
        plt.savefig(save_path)
        plt.close()



    '''
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

        # Find the max x y values
        y_max = max(y_coords)
        x_max = x_coords[y_coords.index(y_max)]

        # Draw max x and y values as text
        plt.text(0.05, 0.95, f"Max X: {max_x}", transform=plt.gca().transAxes, color='black')
        plt.text(0.05, 0.9, f"Max Y: {max_y}", transform=plt.gca().transAxes, color='black')

        # Draw the 'num' value as text
        plt.text(0.05, 0.85, f"Num: {num}", transform=plt.gca().transAxes, color='black')

        # Draw the 'opt' coordinate as text
        opt_x, opt_y = opt
        plt.text(0.05, 0.8, f"Optimal: ({opt_x}, {opt_y})", transform=plt.gca().transAxes, color='black')

        # Set plot title and labels
        plt.title(title)
        plt.xlabel('grid_dim')
        plt.ylabel('efficiency')

        # Display legend
        #plt.legend()

        # Save the plot as an image file
        plt.savefig(save_path)
        plt.close()
    '''

def test_elementwise_mul_half():
    num_elements = 163840*1000
    block_dim = 128
    grid_dim = 240

    ew = ElementWise(num_elements, grid_dim, block_dim)

    title = 'half performance'
    coordinate_list = ew.naive_half_mul(total_num=1500)
    opt = ew.optimized_half_mul()
    ew.draw_performance(num_elements, coordinate_list, 
            opt, title, './half_mul_perf')


def test_elementwise_mul_float():
    num_elements = 163840*1000
    block_dim = 128
    grid_dim = 240

    ew = ElementWise(num_elements, grid_dim, block_dim)

    title = 'float performance'
    coordinate_list = ew.naive_float_mul(total_num=1500)
    opt = ew.optimized_float_mul()
    ew.draw_performance(num_elements, coordinate_list, opt, title, 'float_mul_perf')

def test_elementwise_add_half():
    num_elements = 163840*1000
    block_dim = 128
    grid_dim = 240

    ew = ElementWise(num_elements, grid_dim, block_dim)

    title = 'half performance'
    coordinate_list = ew.naive_half_add(total_num=1500)
    opt = ew.optimized_half_add()
    ew.draw_performance(num_elements, coordinate_list, opt, title, './half_add_perf')

def test_elementwise_add_float():
    num_elements = 163840*1000
    block_dim = 128
    grid_dim = 240

    ew = ElementWise(num_elements, grid_dim, block_dim)

    title = 'float performance'
    coordinate_list = ew.naive_float_add(total_num=1500)
    opt = ew.optimized_float_add()
    ew.draw_performance(num_elements, coordinate_list, opt, title, 'float_add_perf')

def test_draw():
    num_elements = 163840*1000
    block_dim = 128
    grid_dim = 240

    points = [(50, 50), (100, 150), (200, 100), (300, 200), (400, 50)]
    opt = (250, 150)
    title = "Performance Test"
    save_path = "performance_test.png"

    ew = ElementWise(num_elements, grid_dim, block_dim)
    ew.draw_performance(num_elements, points, opt, title, save_path)

test_elementwise_mul_half()
test_elementwise_mul_float()
test_elementwise_add_half()
test_elementwise_add_float()

#test_draw()


