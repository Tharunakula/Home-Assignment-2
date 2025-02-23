**CS5720 Neural network and Deep learning,**

**CRN:-23848**

**Name:- Akula Tharun.**

**student ID:-700759411.**

**Home Assignment 2.**

**Q2. Convolution Operations with Different Parameters**

**Task: Implement Convolution with Different Stride and Padding**

1. perform_convolution(input_matrix, kernel, strides, padding) Function:
   * Takes the input matrix, kernel, strides, and padding as arguments.
   * Reshaping for TensorFlow:
      * input_matrix.reshape((1, 5, 5, 1)) adds a batch dimension (1) and a channel dimension (1) to the input matrix, which is required by tf.nn.conv2d().
      * kernel.reshape((3, 3, 1, 1)) similarly reshapes the kernel.
   * Tensor Conversion:
      * tf.constant() converts the NumPy arrays to TensorFlow tensors.
   * Convolution:
      * tf.nn.conv2d() performs the 2D convolution.
         * strides=[1, strides, strides, 1] sets the strides for the convolution. The first and last strides are 1 (for batch and channel dimensions), and the middle strides are the specified strides value.
         * padding=padding sets the padding type.
   * Reshaping for Output:
      * output_tensor.numpy().reshape(output_tensor.shape[1], output_tensor.shape[2]) removes the batch and channel dimensions from the output tensor and converts it back to a NumPy array.

2. Input Matrix and Kernel:

   * The input matrix and kernel are defined as NumPy arrays.

3. Convolution with Different Parameters:

   * The perform_convolution() function is called with different combinations of strides and padding.
   * The results are stored in a dictionary.

4. Printing Results:
    
    * The output feature maps are printed to the console, along with the corresponding stride and padding values.


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

**Task 2: Implement a Residual Block and ResNet**

1. residual_block() Function:

     * Implements the residual block as described in the problem description.
     * Applies two Conv2D layers with ReLU activation.
     * Uses layers.Add() to add the input tensor to the output of the second Conv2D layer (the skip connection).
     * Applies a ReLU activation after the skip connection.

2. create_resnet_like_model() Function:
    * Creates the ResNet-like model.
    * Uses an initial Conv2D layer.
    * Applies two residual_block() calls.
    * Adds a maxpooling layer after the first convolutional layer, as is common in ResNet architectures.
    * Adds Flatten, Dense (128 neurons), and the output layer.
    * Uses the functional API (models.Model(inputs=inputs, outputs=outputs)) to create the model, which is more flexible than the Sequential API for complex architectures like ResNet.

3. Model Summary:

   * resnet_model.summary() prints the model summary.

4. Input shape and number of classes:

   * The default input shape is (224, 224, 3) and the default number of classes is 10. These can be adjusted when calling create_resnet_like_model() to match your specific dataset.

5. Padding:

    *Padding is set to 'same' for the initial convolutional layer and the residual block convolutional layers. This ensures that the spatial dimensions of the feature maps are preserved, making it easier to add the skip connection.
