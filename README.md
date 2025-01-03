# Epipolar Constraint Guided Differentiable Keypoint Detection and Description

*Sparse local feature matching* techniques have made significant strides in a variety of visual geometry tasks. Among them, the weakly supervised methods have drawn particular attention recently and outperform the fully-supervised counterparts via decoupled describe-then-detect training. However, they often rely on policy gradients for detector training, overlooking keypoint reliabil
ity. Meanwhile, many of the sparse local feature matching methods put more emphasis on accuracy over speed, making them unfriendly for real-time applications. To address these issues, we introduce **the differentiable keypoint extraction** and **the dispersity peak loss** to generate clean score maps and enhance the reliability of the keypoints. The proposed model is trained under **the weakly supervised** fashion by leveraging the epipolar constraint between images. Additionally, **we propose an efficient model that achieves a good balance between accuracy and speed**. Experiments on various public benchmarks show our method achieving higher performance than existing ones.
# Fig.1
![image](https://github.com/FYL0123/WSDK/blob/main/imgs/gflops.jpg)

# Fig.2
![image](https://github.com/FYL0123/WSDK/blob/main/imgs/reli.jpg)

# Training
Download the preprocessed subset of MegaDepth from [CAPS:](https://github.com/qianqianwang68/caps)
