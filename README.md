# Lunar Lander Project
![image](https://github.com/user-attachments/assets/91c2c201-adf6-463b-b648-65fffed49520)

## Overview

The Lunar Lander Project simulates a lunar landing mission with two control strategies: a PID controller and a Deep Q-Network (DQN) agent. The project includes a graphical user interface (GUI) for adjusting simulation parameters such as gravity, wind power, and fuel limits. The goal is to safely land the lunar lander using the selected control strategy.

## Features

- **PID and DQN Controllers**: Test and compare the performance of a PID controller and a trained DQN agent for lunar landing.
- **Customizable Environment Parameters**: Fine-tune settings like gravity, wind power, fuel limits, and malfunctions through an intuitive GUI.
- **Simulation Visualization**: Real-time graphical feedback on the lunar lander's behavior and performance.

## GUI Functionality

The GUI provides the following features:
- **Choose Environment**:
  Select between a Custom or Original environment for the simulation.
  - **Custom Environment**: Users can modify specific parameters, including:
    - **Gravity**: Modify gravity to test controller performance under different conditions.
    - **Wind Power**: Change wind conditions to introduce variability into the simulation.
    - **Fuel Limit**: Set fuel constraints to challenge the controllers.
    - **Malfunction**: Toggle random malfunctions to simulate real-world issues.
- **Controller Selection**:
- Choose between PID and DQN controllers.
- **Number of Iterations**:
- Define the number of simulation runs.
- **Run and Exit Buttons**:
- Start simulations or close the application.

## Installation

To set up the project locally:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/YarinOhayon/Lunar_Lander.git
    cd Lunar_Lander
    ```

2. **Install Dependencies**:
   Ensure Python is installed, then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:
   Launch the GUI by executing:
    ```bash
    python LunarLanderGUI.py
    ```

## Usage

1. **Select a Controller**: Choose PID or DQN from the GUI dropdown menu.
2. **Set Parameters**: Adjust gravity, wind power, fuel limits, and malfunctions as needed.
3. **Start the Simulation**: Click "Run" to begin the simulation with the selected parameters.
4. **Close the Application**: Click "Exit" to terminate the application.

## Code Structure

- **`LunarLanderGUI.py`**: Implements the GUI using Tkinter for user interactions, allowing users to select controllers and adjust simulation parameters.
- **`LunarLanderEnvWrapper.py`**: Wraps the Lunar Lander environment to integrate and manage adjustable parameters like gravity and wind.
- **`DQNAgent.py`**: Contains the implementation of the Deep Q-Network (DQN) agent used to control the lunar lander and make decisions based on the environment state.
- **`LunarLanderPIDController.py`**: Implements the Proportional-Integral-Derivative (PID) controller for managing the lunar lander’s descent.
- **`DQN.py`**: Defines the architecture of the Deep Q-Network (DQN) using PyTorch, including fully connected layers and dropout for reinforcement learning tasks.
- **`PIDController.py`**: Implements a basic PID controller to calculate control outputs based on setpoint and measurement, with methods for computing control actions.
- **`ReplayBuffer.py`**: Provides a replay buffer for storing transitions and sampling mini-batches for training the DQN agent.
- **`requirements.txt`**: Lists the Python packages required to run the project.
- **`README.md`**: Documentation file describing the project, installation, usage, and other relevant information.
