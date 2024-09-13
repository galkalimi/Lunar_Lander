from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit, QCheckBox, QPushButton, QSpinBox, QMessageBox
from PyQt5.QtCore import Qt
import gymnasium as gym
from LunarLanderEnvWrapper import LunarLanderEnvWrapper
from DQNAgent import DQNAgent
from LunarLanderPIDController import LunarLanderPIDController

class LunarLanderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lunar Lander Test Environment")
        self.setStyleSheet("background-color: #000000; color: #FFFFFF;")
        self.running = False

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        title_label = QLabel("Lunar Lander Test Environment")
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        input_frame = QWidget()
        input_layout = QVBoxLayout(input_frame)
        input_frame.setStyleSheet("background-color: #2c2c2c; border: 1px solid #FFFFFF;")
        layout.addWidget(input_frame)

        # Environment selection
        env_layout = QHBoxLayout()
        env_label = QLabel("Choose Environment:")
        env_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.env_choice = QComboBox()
        self.env_choice.addItems(["Original", "Custom"])
        self.env_choice.setCurrentText("Original")
        self.env_choice.currentTextChanged.connect(self.toggle_custom_inputs)
        env_layout.addWidget(env_label)
        env_layout.addWidget(self.env_choice)
        input_layout.addLayout(env_layout)

        # Controller selection
        controller_layout = QHBoxLayout()
        controller_label = QLabel("Choose Controller:")
        controller_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.controller_choice = QComboBox()
        self.controller_choice.addItems(["PID", "DQN"])
        self.controller_choice.setCurrentText("DQN")
        controller_layout.addWidget(controller_label)
        controller_layout.addWidget(self.controller_choice)
        input_layout.addLayout(controller_layout)

        # Gravity setting input
        gravity_layout = QHBoxLayout()
        gravity_label = QLabel("Gravity:")
        gravity_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.gravity_entry = QLineEdit()
        self.gravity_entry.setText("-10")
        gravity_layout.addWidget(gravity_label)
        gravity_layout.addWidget(self.gravity_entry)
        input_layout.addLayout(gravity_layout)

        # Wind power setting input
        wind_layout = QHBoxLayout()
        wind_label = QLabel("Wind Power:")
        wind_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.wind_entry = QLineEdit()
        self.wind_entry.setText("0")
        wind_layout.addWidget(wind_label)
        wind_layout.addWidget(self.wind_entry)
        input_layout.addLayout(wind_layout)

        # Malfunction checkbox
        malfunction_layout = QHBoxLayout()
        malfunction_label = QLabel("Enable Malfunction:")
        malfunction_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.malfunction_check = QCheckBox()
        malfunction_layout.addWidget(malfunction_label)
        malfunction_layout.addStretch()
        malfunction_layout.addWidget(self.malfunction_check)
        malfunction_layout.addStretch()
        input_layout.addLayout(malfunction_layout)

        # Fuel limit input
        fuel_layout = QHBoxLayout()
        fuel_label = QLabel("Fuel Limit:")
        fuel_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.fuel_entry = QLineEdit()
        self.fuel_entry.setText("500")
        fuel_layout.addWidget(fuel_label)
        fuel_layout.addWidget(self.fuel_entry)
        input_layout.addLayout(fuel_layout)

        # Number of iterations input
        iterations_layout = QHBoxLayout()
        iterations_label = QLabel("Number of Iterations:")
        iterations_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.iterations_entry = QSpinBox()
        self.iterations_entry.setValue(1)
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.iterations_entry)
        input_layout.addLayout(iterations_layout)

        # Run button
        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet("background-color: #FFD700; font-size: 14pt; font-weight: bold; color: #000000;")
        self.run_button.clicked.connect(self.run_test)
        layout.addWidget(self.run_button)

        # Exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.setStyleSheet("background-color: #8A2BE2; font-size: 14pt; font-weight: bold; color: #FFFFFF;")
        self.exit_button.clicked.connect(self.close)
        layout.addWidget(self.exit_button)

        self.toggle_custom_inputs()

    def toggle_custom_inputs(self):
        is_custom = self.env_choice.currentText() == "Custom"
        self.gravity_entry.setVisible(is_custom)
        self.wind_entry.setVisible(is_custom)
        self.malfunction_check.setVisible(is_custom)
        self.fuel_entry.setVisible(is_custom)

    def run_test(self):
        self.running = True
        controller = self.controller_choice.currentText()
        num_iterations = self.iterations_entry.value()

        if self.env_choice.currentText() == "Custom":
            gravity = (0, float(self.gravity_entry.text())) if self.gravity_entry.text() else (0, -10)
            wind_power = float(self.wind_entry.text()) if self.wind_entry.text() else 0
            enable_wind = wind_power > 0
            enable_malfunction = self.malfunction_check.isChecked()
            fuel_limit = float(self.fuel_entry.text()) if self.fuel_entry.text() else 500
            env = LunarLanderEnvWrapper(gravity=gravity,
                                        enable_wind=enable_wind, wind_power=wind_power,
                                        enable_fuel=True, fuel_limit=fuel_limit,
                                        enable_malfunction=enable_malfunction, render=True)
        else:
            env = gym.make('LunarLander-v2', render_mode='human')

        if controller == "PID":
            pid_controller = LunarLanderPIDController(env)
            pid_controller.run(num_iterations=num_iterations)
        else:
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            agent = DQNAgent(state_dim, action_dim, env.action_space)
            agent.load_model('dqn_lunarlander_classic.pth')
            agent.run(env, num_episodes=num_iterations)

        self.running = False

    def closeEvent(self, event):
        if self.running:
            QMessageBox.warning(self, "Warning", "Simulation is running. Please wait until it finishes.")
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication([])
    window = LunarLanderGUI()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()