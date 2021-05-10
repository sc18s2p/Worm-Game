Instructions:

1. Install panda3d with a version that is suitable to your Operating System from the panda3d website https://www.panda3d.org/
(The panda3D version I used is in the worm directory)
2. Change Directory to the worm directory
3. Create a Python 3.9 environment with [Conda](https://conda.io/docs/) using the environment file provided:

```bash
conda env create -f environment.yml
```
4. In the command promt type conda activate <name of the virtual environment created>
5. Then type python wormPanda.py to run the game

Game Instructions:
1. Press H for the game to start
2. A,W,D,X,S,Z manipulate the camera
3. J,L turn the worm left and right 
The instructions above are also printed at the top left of the game interface

Game Objective:
Collect the suns to gather points and avoid the moons or else you will lose the game. Careful! The more suns you get the more moons spawn.

