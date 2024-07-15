# Project Documentation
## Initial Goal
Initially this repo was created to create a virtual self-driving agent based purely on Vision by using a classic CNN approach. It was decided to test this on the gamejam game "[Driving Nightmare](https://misterixi.itch.io/driving-nightmare)", since I worked on this game myself and could freely adjust the game to my needs.  
## CNN
As a "normal" CNN, the network was created with a simple convolutional neural network with tensorflow. The input was simply one screenshot of the game, at first with all three color channels and later with only a grayscale image. The output was a one-hot encoded vector of the possible inputs.  
The analog inputs of steering and acceleration were changed to a set of discrete inputs: Steering used steps of 0.1 from -1.0 to 1.0 and thus had 21 possible variants. Accelereation had the same granularity at first, but was later changed to only 3 possible inputs: brake, neutral and accelerate.  
A human would then play the game for a while and an accompanying script would record the input and screenshots. This data was then used to train the network to predict the input based on the screenshot.  
This was a very poor attempt, since the network was not able to infer the current gamestate from just one screenshot. It was very hard to tell what's going on without motion for the model.  
This approach was also very flawed at first, since the model was able to reduce the loss very quickly at first, but performed very poorly ingame. Later it was clear, that a change in the game to display the current input as a visual to easier debug the model was actually very detremental to the learning process. The model just learned to read the input from the image, which led to it basically inputting random input due to variation and ignoring everything else.  

## DQCNN
After the problems with the first approach arose, it was clear that it had to change. Also, since it would fit, it was decided to use a Deep Q-Learning Network with visual input - hence DQCNN. The basic structure stayed the same so far, but the specifics evolved over time.  
Instead of just learning on past data, it is made to switch between learning and playing constantly to improve the reinforcement learning process.  
The model will play the game and record all states, while the user written program provides judgement over the current state and gives out reward (or negative reward to punish). The "experience" won't just be used to learn once though, but will be saved to disk and the learning always randomly pulls from any past experiences to train on. It is biased to pick newer experiences, since they're more likely to be more useful for the training itself.  
This whole change showed a lot of promise initially, but also stagnated after some learning time. It also doesn't seem to be very quick at learning (comparing it to other reinforcement learning agents) and has to be tuned further.  
### NN architecture used for the graphs
You can find the architecture used for the tensorflow version at the time of writing in the [`docs/NN_architecture.md`](./NN_architecture.md) file.  
### Results
The final results are posted in the main readme and you can see a certain progress on both games. The driving is still very much subpar, but the graph proves overall improvement over time. It might just need more learning time, or better tuned parameters.  
## The future of the project
Since there was trouble getting tensorflow to run on the GPU, but pytorch working on the GPU without a problem, the pytorch version was started. It was about 10 times as fast during training, which would speed up finding a better NN architecture and paramters significantly.  
However, unfortunately the pytorch version was not able to learn at all. It seems to learn _something_ at the start, but quickly plateaued and then even gets worse over time again. So there seems to be some fundamental problem with the implementation and needs to be fixed.  
So the next steps would be to fix the pytorch version to further improve the whole DQNN.  
