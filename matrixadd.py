import numpy
import numpy as geek


def main():
    in_arr1 = geek.array([[3, 4], [2, 16]])
    in_arr2 = geek.array([[4, 0.5], [2, 5]])
    print("1st Input array : \n", in_arr1)
    print("2nd Input array : \n", in_arr2)

    out_arr = geek.add(in_arr1, in_arr2)
    print("output added array : \n", out_arr)
    inverse = numpy.linalg.inv(in_arr1)
    print("Inverse of in_arr1: \n", inverse)
    print("identity mat: \n", numpy.mat(in_arr1)*numpy.mat(inverse))


if __name__ == '__main__':
    main()
