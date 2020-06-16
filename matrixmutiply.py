import numpy
import numpy as geek


def main():
    in_arr1 = geek.array([[1, 0], [2, 5], [3, 1]])
    in_arr2 = geek.array([[4, 0.5], [2, .5], [0, 1]])
    print("1st Input array : ", in_arr1)
    print("2nd Input array : ", in_arr2)

    out_arr = geek.multiply(in_arr1, in_arr2)
    print("output multiplied array : ", out_arr)

def main1():
    in_arr1 = geek.array([[1, 3], [4, 0], [2, 1]])
    in_arr2 = geek.array([[1], [5]])
    print("1st Input array : \n", in_arr1)
    print("2nd Input array : \n", in_arr2)

    out_arr = numpy.mat(in_arr1)*numpy.mat(in_arr2)
    print("output multiplied array : \n", out_arr)


if __name__ == '__main__':
    main1()
