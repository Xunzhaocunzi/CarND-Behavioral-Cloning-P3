# **Behavioral Cloning** 

## Writeup Template


My first step was to generate and process the images. Because I found just using the recorded images to train the vehicle was far from enough. The vehicle was just turning around or could not follow the track. Then I had to manually drive the vehicle two laps, one clockwise and the other one counter-clockwise, and then converted all these recorded images to rgb and yuv color spaces. This conversion may help improve the image recoganition under different light conditions. I also flipped the images left and right to double the number of images, and it may help the vehicle learn how to turn left and right. 

The second step is to train the model. Originally I was using Lenet setup, which was very similar to the architecture used in traffic sign project. The result did not meet my target, the vehicle could drive Okay when it was going straight, but it failed when turning. Then I switched to nVIDIA architecture, which was much more complicated than Lenet model. And it worked. 

In addition, I spent some time on the challenge track. The model worked 80% fine except when it approached a very sharp turn. 
