# MachineLearning-Alg
Machine learning excercise using the dataset Fashion-MNIST

# Purpose

The aim of the project is to experimentally study the performance of known classification algorithms and clustering algorithms on a real data set. The dataset that I used is the https://www.tensorflow.org/tutorials/keras/classification .

# Classification
* Nearest Neighbor k-ΝΝ using
  * Euclidean distance
  * Cosine distance
(assuming k nearest neighbors (values that i tried k = 1, 5, 10))
* Neural networks with sigmoid activation function in every neuron
  * With 1 hidden layer and K hidden neurons and
  * With 2 hidden layers with K1 neuron in the one hidden layer and K2 neurons in
  the other
I used the Stochastic Gradient Descent optimization method to train them. The output of the
network consists of 10 neurons where, using the softmax activation function, the
probability that a data (image) belongs to each category will be calculated.
Indicative values of the number of neurons are: K = 500, K1 = 500, K2 = 200.
* Support Vector Machines (SVM) using
  * Linear kernel
  * Gaussian kernel (testing different values of its parameter)
  * Cosine kernel

The evaluation of the performance of the methods is done with the following evaluation measures on the whole testing (testing)

![image](https://user-images.githubusercontent.com/25778156/122991543-d158e900-d3ad-11eb-9eda-210a2a25b9a2.png)


# Clustering

The data type that I used to represent each image was a vector with length 28x28 = 784 where each component represents the brightness value of each pixel of the image

* K-means clustering algorithm from **scratch** (we used K = 10 clusters) 
  * Euclidean distance
  * Manhattan distance
  * cosine distance

To compare the clustering methods I used the following two evaluation measures:
* **Purity**
  The category of each cluster is determined, after the end of the clustering, by the majority of the actual category among the members of the group.
  Then purity is calculated by measuring the percentage of correctly classified data.
* **F-measure**  
  For each cluster, after defining the majority category as a cluster category (as in the previous measure), I found TP (true positive), FP (false positive) and FN 
  (false negative) and then the F1-score. Finally, the evaluation of the clustering method results from the sum of the F-measures for each cluster.
  ![image](https://user-images.githubusercontent.com/25778156/122992676-0f0a4180-d3af-11eb-9d9a-912c33ea176b.png)

  
