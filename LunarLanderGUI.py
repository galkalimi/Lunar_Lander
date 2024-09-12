import tkinter as tk
from tkinter import Canvas, ttk
from LunarLanderEnvWrapper import LunarLanderEnvWrapper
from DQNAgent import DQNAgent
from LunarLanderPIDController import LunarLanderPIDController
import threading

class LunarLanderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lunar Lander Test Environment")
        self.root.configure(bg="#000000")  # Black background

        # Style configurations
        style = ttk.Style()
        style.configure("DarkFrame.TFrame", background="#2c2c2c", borderwidth=1, relief="solid")
        style.configure("TLabel", foreground="#FFFFFF", background="#000000")  # White text
        style.configure("TEntry", fieldbackground="#4a4a4a", foreground="#000000")  # Gray background for entry, black text
        style.configure("TCheckbutton", foreground="#FFFFFF", background="#000000")  # White text for checkboxes

        # Title label with white color
        title_label = ttk.Label(self.root, text="Lunar Lander Test Environment", font=("Helvetica", 18, "bold"),
                                foreground="#FFFFFF", background="#000000")  # White for title text
        title_label.grid(row=0, column=0, columnspan=2, pady=20)

        # Frame for input controls with a modern dark gray border
        self.input_frame = ttk.Frame(self.root, padding="10", style="DarkFrame.TFrame")
        self.input_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Controller selection
        ttk.Label(self.input_frame, text="Choose Controller:", font=("Helvetica", 12, "bold")).grid(
            row=1, column=0, padx=10, pady=5, sticky="w")
        self.controller_var = tk.StringVar()
        self.controller_choice = ttk.Combobox(self.input_frame, textvariable=self.controller_var, values=["PID", "DQN"])
        self.controller_choice.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.controller_choice.set("DQN")

        # Gravity setting
        ttk.Label(self.input_frame, text="Gravity:", font=("Helvetica", 12, "bold")).grid(
            row=2, column=0, padx=10, pady=5, sticky="w")
        self.gravity_var = tk.DoubleVar()
        self.gravity_entry = ttk.Entry(self.input_frame, textvariable=self.gravity_var)
        self.gravity_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.gravity_var.set(-10)

        # Wind setting
        ttk.Label(self.input_frame, text="Wind Power:", font=("Helvetica", 12, "bold")).grid(
            row=3, column=0, padx=10, pady=5, sticky="w")
        self.wind_var = tk.DoubleVar()
        self.wind_entry = ttk.Entry(self.input_frame, textvariable=self.wind_var)
        self.wind_entry.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        self.wind_var.set(0)


        # Malfunction setting
        ttk.Label(self.input_frame, text="Enable Malfunction:", font=("Helvetica", 12, "bold")).grid(
            row=4, column=0, padx=10, pady=5, sticky="w")
        malfunction_check_frame = ttk.Frame(self.input_frame, style="DarkFrame.TFrame", borderwidth=0, relief='flat')
        malfunction_check_frame.grid(row=4, column=1, padx=10, pady=5, sticky="nsew")
        malfunction_check_frame.grid_columnconfigure(0, weight=1)
        self.malfunction_var = tk.BooleanVar()
        style.configure("CenteredCheck.TCheckbutton", background="#2c2c2c", foreground="#FFFFFF")  # Dark gray background, white text
        self.malfunction_check = ttk.Checkbutton(malfunction_check_frame, variable=self.malfunction_var, style="CenteredCheck.TCheckbutton")
        self.malfunction_check.grid(row=0, column=0, padx=0, pady=0, sticky="")

        # Fuel setting
        ttk.Label(self.input_frame, text="Fuel Limit:", font=("Helvetica", 12, "bold")).grid(
            row=5, column=0, padx=10, pady=5, sticky="w")
        self.fuel_var = tk.DoubleVar()
        self.fuel_entry = ttk.Entry(self.input_frame, textvariable=self.fuel_var)
        self.fuel_entry.grid(row=5, column=1, padx=10, pady=5, sticky="ew")
        self.fuel_var.set(200)

        # Number of iterations
        ttk.Label(self.input_frame, text="Number of Iterations:", font=("Helvetica", 12, "bold")).grid(
            row=6, column=0, padx=10, pady=5, sticky="w")
        self.iterations_var = tk.IntVar()
        self.iterations_entry = ttk.Entry(self.input_frame, textvariable=self.iterations_var)
        self.iterations_entry.grid(row=6, column=1, padx=10, pady=5, sticky="ew")
        self.iterations_var.set(1)

        # Run Button with yellow background
        self.run_button = tk.Button(self.root, text="Run", command=self.start_test_thread,
                                    bg="#FFD700", font=("Helvetica", 14, "bold"), relief="ridge", bd=4, fg="#000000")  # Yellow with black text
        self.run_button.grid(row=7, column=0, columnspan=2, pady=15, padx=20, sticky="ew")

        # Exit Button with purple background
        self.exit_button = tk.Button(self.root, text="Exit", command=self.on_exit,
                                     bg="#8A2BE2", font=("Helvetica", 14, "bold"), relief="ridge", bd=4, fg="#FFFFFF")  # Purple with white text
        self.exit_button.grid(row=8, column=0, columnspan=2, pady=10, padx=20, sticky="ew")

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
        wind_power = self.wind_var.get()
        enable_malfunction = self.malfunction_var.get()  # Not implemented yet
        fuel_limit = self.fuel_var.get()
        num_iterations = self.iterations_var.get()

        # Initialize the environment with the selected settings
        env = LunarLanderEnvWrapper(gravity=gravity,
                                    enable_wind=True, wind_power=wind_power,
                                    enable_fuel=True, fuel_limit=fuel_limit,
                                    enable_malfunction=enable_malfunction, render=True)

        if controller == "PID":
            pid_controller = LunarLanderPIDController(env)
            pid_controller.run(self.stop_event, num_iterations)  # Pass stop_event to PID Controller
        else:
            # For DQN
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            agent = DQNAgent(state_dim, action_dim, env.action_space)
            agent.load_model('dqn_lunarlander_classic.pth')
            agent.run(env, num_iterations, self.stop_event)


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