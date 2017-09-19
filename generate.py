from matplotlib import pyplot as plt
import numpy as np
import argparse

scale_factor = 10   # The scale factor which will be used to normalize the range of x

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='./data1.dat', dest='file_name', help='the destination path of the generated points')
    parser.add_argument('--format', type=str, default='sin', dest='format', help='the format of target function')
    parser.add_argument('--num', type=int, default=10, dest='point_num', help='the number of sample points')
    args = parser.parse_args()

    # Generate points
    with open(args.file_name, 'w') as f:
        x_coordinates = np.linspace(-scale_factor, scale_factor, args.point_num)
        if args.format == 'sin':
            y_coordinates = np.sin(x_coordinates)
        elif args.format == 'cos':
            y_coordinates = np.cos(x_coordinates)
        elif args.format == 'tan':
            y_coordinates = np.tan(x_coordinates)
        else:
            print('invalid target function character...')
            exit()
        x_coordinates /= scale_factor
        y_coordinates += np.random.normal(-0.1, 0.1, args.point_num)

        # Write into files
        for i in range(len(x_coordinates)):
            f.write(str(x_coordinates[i]) + ', ' + str(y_coordinates[i]) + '\n')