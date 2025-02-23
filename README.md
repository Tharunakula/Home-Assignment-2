**CS5720 Neural network and Deep learning,

CRN:-23848

Name:- Akula Tharun.

student ID:-700759411.

Home Assignment 2.
**

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
