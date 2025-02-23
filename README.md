**CS5720 Neural network and Deep learning,**

**CRN:-23848**

**Name:- Akula Tharun.**

**student ID:-700759411.**

**Home Assignment 2.**

**Q2. Convolution Operations with Different Parameters**

**Task: Implement Convolution with Different Stride and Padding**



**Q3. CNN Feature Extraction with Filters and Pooling**

**Task 1: Implement Edge Detection Using Convolution** 

1. Import Libraries:

  * cv2 (OpenCV): For image loading and Sobel filter application.
  * numpy: For numerical operations (especially with image arrays).
  * matplotlib.pyplot: For displaying the images.
 
2. edge_detection_sobel(image_path) Function: 

     Load Image:
   
            * cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) loads the image in grayscale mode.
            * Error handling is added to ensure that the file exists.
            
4. Apply Sobel Filter:

      * cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) applies the Sobel filter in the x-direction.
      * cv2.CV_64F specifies the output image depth (64-bit float).
      * 1,0 specifies that the derivative is taken in the x direction.
      * ksize=3 is the kernel size.
      * cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) applies the Sobel filter in the y-direction.
      * 0,1 specifies that  the derivative is taken in the y direction.
      * np.absolute() takes the absolute value of the result.
      * np.uint8() converts the result back to 8-bit unsigned integer (for display).

5. Display Images:
   
         * matplotlib.pyplot is used to display the original image, the Sobel X result, and the Sobel Y result in a single figure.

Replace 'your_image.jpg' with the actual path to your image file.
Call the edge_detection_sobel() function.

**Task 2: Implement Max Pooling and Average Pooling**

1. Import Libraries:

    * tensorflow (as tf): For TensorFlow/Keras operations.
    * numpy (as np): For creating the random matrix.

 2. pooling_demo() Function:

      * Create Input Matrix:

           * np.random.randint(0, 10, size=(1, 4, 4, 1)) creates a 4x4 matrix with random integers between 0 and 9.
           * The size parameter is (1, 4, 4, 1):
              * 1: Batch size (one image).
              * 4: Height.
              * 4: Width.
              * 1: Number of channels (grayscale).
           * tf.constant() converts the NumPy array to a TensorFlow tensor.

3.Max Pooling:

    * tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor) applies Max Pooling.
    * pool_size=(2, 2): The pooling window is 2x2.
    * strides=(2, 2): The window moves 2 pixels in each direction.
    * padding='valid' means no padding is applied.

4.Average Pooling:

    * tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor) applies Average Pooling.
    * The parameters are the same as Max Pooling.

5. Print Results:

       * input_matrix[0, :, :, 0] removes the batch and channel dimensions to print the original matrix.
       * max_pool.numpy()[0, :, :, 0] and avg_pool.numpy()[0, :, :, 0] extract the NumPy arrays from the TensorFlow tensors and remove the batch and channel dimensions for printing.

**Q4. Implementing and Comparing CNN Architectures**

**Task 1: Implement AlexNet Architecture**

1. create_simplified_alexnet() Function:

    * Encapsulates the model creation for better organization and reusability.
    * Takes input_shape and num_classes as arguments, making the model more flexible.

2. padding='same':

   * I've added padding='same' to the Conv2D layers (except the first one) where the kernel size is (3, 3). Without padding, the spatial dimensions of the feature maps shrink after each convolution, which can lead to a smaller-than-expected output shape before the Flatten layer. padding='same' ensures that the output feature maps have the same spatial dimensions as the input feature maps for those layers, which is more in line with the usual behavior of AlexNet-like architectures.

3. Model Summary:

   * alexnet_model.summary() prints a detailed summary of the model's architecture, including the layer types, output shapes, and number of parameters.

4. Input shape:

   * The default input shape is set to (227, 227, 3) which is a common input size for AlexNet. You can change this if your images have a different size.

5. Num Classes:

  * The default number of classes is set to 10. If you are working on a classification task with a different number of classes, you can modify this parameter.
