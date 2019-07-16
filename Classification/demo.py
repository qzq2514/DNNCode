import numpy as np
import time

start_time=time.time()
arr=[[2.29058266e-08, 8.90816381e-08, 3.21978529e-04, 1.78394245e-07,
        3.07489529e-07, 4.48791941e-11, 7.98808575e-09, 1.45710444e-09,
        9.20440798e-05, 2.02744210e-10, 3.46077373e-03, 3.37084231e-04,
        2.63022853e-10, 5.48258363e-08, 9.16976546e-07, 2.98219129e-06,
        4.01643345e-12, 9.13220923e-04, 6.75789771e-11, 3.24528571e-10,
        1.07656626e-06, 4.23044320e-08, 9.39865172e-01, 1.13093066e-07,
        1.06209645e-10, 2.48862002e-02, 3.70230235e-11, 3.01160663e-02,
        6.49365423e-11, 6.93358981e-10, 2.09636641e-09, 5.94107785e-09,
        1.31337924e-06, 2.04677136e-07, 3.92830346e-09, 1.27938575e-08]]

arr=np.array(arr)
print(arr[:,3:5])
label_id=np.argmax(arr,1)
print(label_id)
print(time.time()-start_time)
print("---------------")

st="resNet_Digit"

print(st=="resNet_Digit")

print("---------------")

arr=[2,2]
print(arr==[2,2])
print("----------------------")


def channel_shuffle1(inputs, group_num):
        N, H, W, C = inputs.shape
        inputs_reshaped = np.reshape(inputs, [-1, H, W, group_num, C // group_num])
        inputs_transposed = np.transpose(inputs_reshaped, [0, 1, 2, 4, 3])
        result = np.reshape(inputs_transposed, [-1, H, W, C])
        return result

def channel_shuffle2(inputs, group_num):
        N, H, W, C = inputs.shape
        for n in range(N):
           for h in range(H):
              for w in range(W):
                print(inputs[n,h,w,:])
                # inputs[n,h,w,:]=np.random.shuffle(inputs[n,h,w,:])
        # return inputs

arr_shuffle1=np.arange(1,37).reshape([1,2,3,6])
print(arr_shuffle1)
print("------channel_shuffle1------")
arr_shuffle2=channel_shuffle1(arr_shuffle1,2)
print(arr_shuffle2)
print("------channel_shuffle2------")
# arr_shuffle3=channel_shuffle2(arr_shuffle1,2)
# print(arr_shuffle3)

print("-----------------")
a=[11,22,33,44,55,66]
b="ABCDEF"
for ind ,(aa,bb) in enumerate(zip(a,b)):
    print(ind,aa,bb)