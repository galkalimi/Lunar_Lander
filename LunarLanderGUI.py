import tkinter as tk
from tkinter import ttk
from LunarLanderEnvWrapper import LunarLanderEnvWrapper
from DQNAgent import DQNAgent
from LunarLanderPIDController import LunarLanderPIDController
import threading
import sys

class LunarLanderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lunar Lander GUI")
        
        # Title label
        ttk.Label(self.root, text="Lunar Lander Test Environment", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Controller selection
        ttk.Label(self.root, text="Choose Controller:").grid(row=1, column=0, padx=10, pady=5)
        self.controller_var = tk.StringVar()
        self.controller_choice = ttk.Combobox(self.root, textvariable=self.controller_var, values=["PID", "DQN"])
        self.controller_choice.grid(row=1, column=1, padx=10, pady=5)
        self.controller_choice.set("PID")

        # Gravity setting
        ttk.Label(self.root, text="Gravity:").grid(row=2, column=0, padx=10, pady=5)
        self.gravity_var = tk.DoubleVar()
        self.gravity_entry = ttk.Entry(self.root, textvariable=self.gravity_var)
        self.gravity_entry.grid(row=2, column=1, padx=10, pady=5)
        self.gravity_var.set(-10)  # Default gravity

        # Wind setting
        ttk.Label(self.root, text="Enable Wind:").grid(row=3, column=0, padx=10, pady=5)
        self.wind_var = tk.BooleanVar()
        self.wind_check = ttk.Checkbutton(self.root, variable=self.wind_var)
        self.wind_check.grid(row=3, column=1, padx=10, pady=5)

        # Malfunction setting
        ttk.Label(self.root, text="Enable Malfunction:").grid(row=4, column=0, padx=10, pady=5)
        self.malfunction_var = tk.BooleanVar()
        self.malfunction_check = ttk.Checkbutton(self.root, variable=self.malfunction_var)
        self.malfunction_check.grid(row=4, column=1, padx=10, pady=5)

        # Fuel setting
        ttk.Label(self.root, text="Fuel Limit:").grid(row=5, column=0, padx=10, pady=5)
        self.fuel_var = tk.DoubleVar()
        self.fuel_entry = ttk.Entry(self.root, textvariable=self.fuel_var)
        self.fuel_entry.grid(row=5, column=1, padx=10, pady=5)
        self.fuel_var.set(1000)  # Default fuel

        # Number of iterations
        ttk.Label(self.root, text="Number of Iterations:").grid(row=6, column=0, padx=10, pady=5)
        self.iterations_var = tk.IntVar()
        self.iterations_entry = ttk.Entry(self.root, textvariable=self.iterations_var)
        self.iterations_entry.grid(row=6, column=1, padx=10, pady=5)
        self.iterations_var.set(100)  # Default number of iterations

        # Test Button
        self.test_button = ttk.Button(self.root, text="Run Test", command=self.start_test_thread)
        self.test_button.grid(row=7, column=0, columnspan=2, pady=10)

        # Exit Button
        self.exit_button = ttk.Button(self.root, text="Exit", command=self.on_exit)
        self.exit_button.grid(row=8, column=0, columnspan=2, pady=10)

        # Manage threading
        self.test_thread = None
        self.stop_event = threading.Event()

    def start_test_thread(self):
        if self.test_thread and self.test_thread.is_alive():
            # Optionally handle stopping or interrupting the thread if needed
            self.stop_event.set()
        self.stop_event.clear()
        self.test_thread = threading.Thread(target=self.run_test)
        self.test_thread.start()

    def run_test(self):
        # Retrieve user choices
        controller = self.controller_var.get()
        gravity = (0, self.gravity_var.get())
        enable_wind = self.wind_var.get()
        malfunction = self.malfunction_var.get()  # Not implemented yet
        fuel_limit = self.fuel_var.get()
        num_iterations = self.iterations_var.get()

        # Initialize the environment with the selected settings
        env = LunarLanderEnvWrapper(gravity=gravity, enable_wind=enable_wind)
        env.fuel_limit = fuel_limit

        if controller == "PID":
            pid_controller = LunarLanderPIDController(env)
            pid_controller.run(self.stop_event, num_iterations=num_iterations)  # Pass stop_event to PID Controller
        else:
            # For DQN
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            agent = DQNAgent(state_dim, action_dim, env.action_space)
            agent.load_model('dqn_lunarlander_classic.pth')
            agent.epsilon = 0.0  
            # Reset environment
            state, info = env.reset()
            done = False
            total_reward = 0
            for i in range(num_iterations):
                if self.stop_event and self.stop_event.is_set():
                    break
                for _ in range(200):  # Limit steps per episode
                    action = agent.select_action(state)
                    next_state, reward, done, truncated, info = env.step(action)
                    state = next_state

                    # Render the environment after every step
                    env.render()

                    if done or truncated:
                        break

    def on_exit(self):
        if self.test_thread and self.test_thread.is_alive():
            self.stop_event.set()
            self.test_thread.join()  # Wait for the test thread to finish
        self.root.quit()  # Exit the Tkinter main loop

def main():
    root = tk.Tk()
    app = LunarLanderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
