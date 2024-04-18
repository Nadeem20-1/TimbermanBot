Timberman Bot - In Progress
=
Overview
-
To understand Object Detection and YOLOv8 more, I pursued into a basic project regarding a game called timberman. 

Within this game, you must continuously chop by either left/right clicking, or by hitting your left/right arrow keys. Your main objective is to avoid branches by moving to the opposite side of the tree, going as fast as you can.

The objective of this project was to build a bot which would be able to detect objects within the game. There are four objects that are to be detected: Timberman, Left Branch, Right Branch, and finally Game Over. The bot will continuously take screenshots of the game, providing an input for the model. The model will then spit out where it detects any objects that it was trained to detect. In my testing, almost every object was correctly detected and predicted, with an accuracy of around 90% and higher. 

In my program, there is a continuous loop, that will function to take a screenshot, calculate the Intersection of Union value (intersection of two bounding boxes),  and switch accordingly to either the left or right side. Because there will only be one intersection, which is timberman and a branch, we only need to check for one intersection. 

As it stands now, the program functions correctly, getting a high score of around 800.

File Structure:
-
To run the file, ensure you have timberman installed. Install the required libraries within the requirements.txt using 'pip install -r requirements.txt'.
Open up 'timbermanbot.py', and run it. View where the output window is grabbing the screenshots from, and move timberman within said window.
Play, and let it run! Ensure the game window is selected after the CV2 window pops up. Within the game settings, set the game to 'Windowed'. Resolution shouln't matter, but I know it works with 1800x1169.

timberman_old.py was an attempt at running the game controls and screenshotting separately on different processes. This works in terms of speeding up the application itself, but an issue arises when it comes to variable access and changes. There will be times where timberman and branch intersection can occur at the exact same time as the program inputs a left/right key, KILLING timberman in the process! Not good! I tried to fix this issue by introducing locks, but I am at a standstill. Please message me if you have any suggestions. The code is also slightly outdated, as I just worked on the main timberman.py.

The model creator can be found within timberman_model_creator.ipynb. There, you can see the process I took to get the current model(s).

The models can be found within /detect/train19/weights.


Problems Encountered:
-
There were many problems that I encountered throughout my time building this application. It's not even finished yet, I just had to share what I have so far!

The biggest issue was performance. If you look up a timberman bot on youtube, you will see how fast they can be! They use a different approach to detecting objects and moving timberman accordingly. While their approaches are FAR superior to what I am attempting, I can adapt my approach to other games that may have a different approach in winning said game, allowing for easier integration. For an example, I will be working on a bot for Wizard101 that will allow me to perform specific actions, depending on what's on the screen, allowing for potential AFK farming on tedious tasks.

To try and combat the performance issue, I tried to move the functions for left/right click within a loop, and screenshots within a loop, within two different processes. This surprisingly worked out great if you look at them separately, but in practice it isn't as good as anticipated. With this, the biggest issue comes from variables being changed within the same time as another function performs an action depending on the same variable! I explained earlier that exact issue, and I had to take a step back, and use the approach for putting everything on one process.

There is also a big issue when it comes to how resource intensive this can be. Taking multiple screenshots within a few seconds can really tax your device. In my case, I've needed to charge my laptop if I'm doing a session that's more than 6 hours. 

To-Do:
-
- Include support for Windows environments (Currently only tested for Apple devices)
- Optimize Apple version (Faster way to screenshot, code optimizations)
- Create user-friendly GUI
- Update model to include multiple maps (Currently doesn't support one map, while being sketchy in some others)

External links
-
To learn more about the potential of YOLOv8, visit https://docs.ultralytics.com/
