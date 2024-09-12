# Lunar Lander Project

## Overview

This project implements a Lunar Lander simulation environment with two controllers: a PID controller and a Deep Q-Network (DQN) agent. The user can interact with the environment using a graphical user interface (GUI) to adjust various simulation parameters such as gravity, wind power, and fuel limits. The primary goal is to land the lunar lander safely on the surface using one of the trained models.

## Features

- **PID and DQN Controllers**: Control the lunar lander using a PID controller or a trained DQN agent.
- **Adjustable Environment Parameters**: Modify settings like gravity, wind power, fuel limit, and malfunction settings through an intuitive GUI.
- **Simulation Visualization**: Visual feedback of the lunar lander's movements in the environment.
- **Threaded Execution**: Run simulations on a separate thread to keep the GUI responsive.

## GUI Functionality

The GUI allows the user to:

- **Controller Selection**: Choose between PID and DQN controllers.
- **Adjust Environment Parameters**:
  - **Gravity**: Adjust the gravity to test the robustness of the controllers.
  - **Wind Power**: Modify wind conditions to add randomness to the simulation.
  - **Fuel Limit**: Set a fuel constraint to challenge the controller's efficiency.
  - **Malfunction**: Enable or disable random malfunctions during the simulation.
  - **Number of Iterations**: Specify how many simulation runs to perform.
- **Run and Exit Buttons**: Start the simulation with the chosen settings or exit the application.

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd lunar-lander-project
    ```

2. **Install Dependencies**:

   Ensure Python is installed on your system. Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:

   To launch the GUI and start interacting with the lunar lander environment, run:
    ```bash
    python main.py
    ```

## Usage

1. **Choose a Controller**: Select either PID or DQN from the dropdown menu.
2. **Adjust Parameters**: Set the values for gravity, wind power, fuel limits, and enable or disable malfunctions.
3. **Run the Simulation**: Click the "Run" button to start the simulation with the specified settings.
4. **Exit**: Click the "Exit" button to close the application.

## Code Structure

- **`LunarLanderGUI.py`**: Contains the GUI implementation using Tkinter for user interactions.
- **`LunarLanderEnvWrapper.py`**: Wraps the Lunar Lander environment to integrate adjustable parameters.
- **`DQNAgent.py`**: Implements the DQN agent used to control the lunar lander.
- **`LunarLanderPIDController.py`**: Implements the PID controller used to control the lunar lander.

## Future Enhancements

- Implement malfunction handling within the controllers to simulate real-world unpredictability.
- Add more advanced reinforcement learning algorithms.
- Improve visualization with more detailed feedback on each simulation run.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaboration, please contact [Your Name] at [Your Email].