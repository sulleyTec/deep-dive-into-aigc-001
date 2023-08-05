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
        x, y = opt
        plt.text(0.95, 0.05, f"oneflow: ({x=}, {y=:.2f})", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')

        # Draw max x and y values as text
        plt.text(0.95, 0.1, f"{max_x=}", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')
        plt.text(0.95, 0.15, f"{max_y=:.2f}", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')

        # Draw the 'num' value as text
        plt.text(0.95, 0.2, f"#elements: {num}", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')

        # Save the image
        plt.savefig(save_path)
        plt.close()


num_elements_list = [163840*1000, 32*1024*1024]

def test_elementwise_mul_half(num_elements):
    #num_elements = 163840*1000
    block_dim = 128
    grid_dim = 240

    ew = ElementWise(num_elements, grid_dim, block_dim)

    title = f'half performance with {str(num_elements)} elements'
    coordinate_list = ew.naive_half_mul(total_num=1500)
    opt = ew.optimized_half_mul()
    ew.draw_performance(num_elements, coordinate_list, 
            opt, title, f'./ew_mul_half_{str(num_elements)}')

def test_elementwise_mul_float(num_elements):
    #num_elements = 163840*1000
    block_dim = 128
    grid_dim = 240

    ew = ElementWise(num_elements, grid_dim, block_dim)

    title = f'float performance with {str(num_elements)} elements'
    coordinate_list = ew.naive_float_mul(total_num=1500)
    opt = ew.optimized_float_mul()
    ew.draw_performance(num_elements, coordinate_list, 
            opt, title, f'ew_mul_float_{str(num_elements)}')

def test_elementwise_add_half(num_elements):
    #num_elements = 163840*1000
    block_dim = 128
    grid_dim = 240

    ew = ElementWise(num_elements, grid_dim, block_dim)

    title = f'half performance with {str(num_elements)} elements'
    coordinate_list = ew.naive_half_add(total_num=1500)
    opt = ew.optimized_half_add()
    ew.draw_performance(num_elements, coordinate_list, 
            opt, title, f'./ew_add_half_{str(num_elements)}')

def test_elementwise_add_float(num_elements):
    #num_elements = 163840*1000
    block_dim = 128
    grid_dim = 240

    ew = ElementWise(num_elements, grid_dim, block_dim)

    title = f'float performance with {str(num_elements)} elements'
    coordinate_list = ew.naive_float_add(total_num=1500)
    opt = ew.optimized_float_add()
    ew.draw_performance(num_elements, coordinate_list, 
            opt, title, f'ew_add_float_{str(num_elements)}')

def test_draw(num_elements):
    #num_elements = 163840*1000
    block_dim = 128
    grid_dim = 240

    points = [(50, 50), (100, 150), (200, 100), (300, 200), (400, 50)]
    opt = (250, 150)
    title = "Performance Test"
    save_path = "performance_test.png"

    ew = ElementWise(num_elements, grid_dim, block_dim)
    ew.draw_performance(num_elements, points, opt, title, save_path)

for n in num_elements_list:
    test_elementwise_mul_half(n)
    test_elementwise_mul_float(n)
    test_elementwise_add_half(n)
    test_elementwise_add_float(n)

#test_draw()


