# MPC-Nav
## Launching MPC-Nav
Launch simulation instance:
```
roslaunch gazebo_sim people_following_sim.launch
```
Launch fuzzy controller:
```
roslaunch mpc_people_following fuzzy_controller.launch
```
Launch MPC controller:
```
roslaunch mpc_people_following mpc_controller.launch
```
## Additional Information
Launch actor controller:
```
rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/target/cmd_vel
```

Launch the Docker instance:
```
sudo rocker --volume /home/mantis:/home/mantis --nvidia --x11 --privileged  palroboticssl/tiago_tutorials:noetic
```
Dependencies:
```
pip install casadi
pip install scikit-fuzzy
pip install packaging
```
