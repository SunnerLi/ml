import numpy as np

def getCoordinateList(xmin, xmax, ymin, ymax, period = 1):
    # Validate the parameters
    x_num = (xmax - xmin) // period + 1
    y_num = (ymax - ymin) // period + 1
    if xmin > xmax:
        xmax, xmin = xmin, xmax
    if ymin > ymax:
        ymax, ymin = ymin, ymax

    # Get coordinate grids
    x = np.linspace(-10, 4, x_num)
    y = np.linspace(-2, 6, y_num)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_list = np.ndarray([x_num * y_num, 2])
    print(np.shape(grid_x))

    # Fill into list
    counter = 0
    for x, y in zip(np.reshape(grid_x, [-1]), np.reshape(grid_y, [-1])):
        grid_list[counter][0] = x
        grid_list[counter][1] = y
        counter += 1
    return grid_list

# _list = np.asarray([ int(x[:-1]) for x in open('border1.out', 'r').readlines()])
# print(_list, type(_list), type(_list[0]))

a = np.asarray([True, False, True])
print(np.invert(a))