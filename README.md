# Understanding inertial odometry with DNN

In navigation, deep learning for inertial odometry (IO) has recently been investigated using data from a low-cost IMU only. 
The measurement of noise, bias, and some errors from which IO suffers is estimated with a deep neural network (DNN) to achieve more accurate pose estimation.
While numerous studies on the subject highlighted the performances of their approach, the behavior of data-driven IO with DNN has not been clarified.
Through this work, we introduced the remaining problems in IO and hope our work will promote further research.

Further reading : https://ieeexplore.ieee.org/document/9366470 

The code for training and testing one version of the neural network introduced in the paper is available.
Kitti dataset was used for this version of the code.


To use :

```bash

$ ./main_kitty

```

Two options are available in main_kitty:

	- train_veloNet
	- test_veloNet


Set the value to 1 to launch the desired function.

One example of a Neural network was trained and saved in the folder ./saved_NN

