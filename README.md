# Anomaly detection in Video Feeds
The official repository of our ML Project Anomaly detection in Video Feeds.

### Team Name
Team Gabru

### Team Members
[Aishik Chakraborty](https://github.com/aishikchakraborty) 13CS30041

[Ashish Sharma](https://github.com/ash-shar) 13CS30043

[Chinmaya Pancholi](https://github.com/chinmayapancholi13) 13CS30010

[Jatin Arora](https://github.com/jatinarora2702) 13CS10057

[Jeenu Grover](https://github.com/groverjeenu) 13CS30042

[Prabhat Agarwal](https://github.com/prabhat1081) 13CS10060

### Presentation

The presentation can be found here ([Slides](https://github.com/cs60050/TeamGabru/blob/master/ML-Anamoly-Detection-TeamGabru.pptx), [pdf](https://github.com/cs60050/TeamGabru/blob/master/Anomaly-Detection-in-Videos.pdf))

### Approaches

**1. Using Optical Flow:** Extracting spatial-temporal features using optical flow and training classifiers for detecting anomaly

**2. Using AlexNet:** Alexnet contains eight learned layers, five convolutional and three fully-connected layers. Pretrained Alexnet is applied to each frame of the video. Then we take the output of the fc7 layer which gives us a 4096 dimensional vector. 

**3. Time Series Analysis:** Standard statistical techniques for anomaly detection in the feature space we obtain from AlexNet.

**4. Topic Modelling:** A generative model of typical behavior is learned using good discriminative features, and then abnormal behaviors (outliers) are detected and classified as those that are badly explained by the learned model.

### References

1. Rousseeuw, P. and Leroy, A.: 1996, Robust Regression and Outlier Detection. John Wiley & Sons., 3 edition
2. Torr, P. H. S. and Murray, D. W.: 1993, ‘Outlier Detection and Motion Segmentation’. In: Proceedings of SPIE
3. Fawcett, T. and Provost, F. J.: 1999, ‘Activity Monitoring: Noticing Interesting Changes in Behavior’. In: Proceedings of the 5th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 53–62
4. Japkowicz, N., Myers, C., and Gluck M. A.: 1995, ‘A Novelty Detection Approach to Classification’. In: Proceedings of the 14th International Conference on Artificial Intelligence (IJCAI-95). pp. 518–523
5. http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
6. Saligrama, Venkatesh, and Zhu Chen. "Video anomaly detection based on local statistical aggregates." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.
7. https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet
8. T. Hofmann. Unsupervised learning by probability latent semantic analysis. Machine Learning, 42:177–196, 2001
9. Varadarajan, Jagannadan, and Jean-Marc Odobez. "Topic models for scene analysis and abnormality detection." Computer Vision Workshops (ICCV Workshops), 2009 IEEE 12th International Conference on. IEEE, 2009.
