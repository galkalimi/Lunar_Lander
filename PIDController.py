class PIDController:
    """
    A Proportional-Integral-Derivative (PID) controller.

    This class implements a basic PID controller that calculates an output value
    based on a setpoint and a measurement. The controller aims to minimize the error
    between the setpoint and the measurement by adjusting the output.

    Attributes:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        setpoint (float): The desired setpoint that the controller aims to reach.
        integral (float): The accumulated integral of the error over time.
        previous_error (float): The error from the previous step, used to calculate the derivative term.

    Methods:
        compute(measurement, dt):
            Computes the control output based on the current measurement and time step.
    
    Example:
        pid = PIDController(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=10)
        output = pid.compute(measurement=8, dt=0.1)
    """

    def __init__(self, Kp, Ki, Kd, setpoint=0):
        """
        Initializes the PIDController with the given gains and setpoint.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            setpoint (float, optional): The desired setpoint. Defaults to 0.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0

    def compute(self, measurement, dt):
        """
        Computes the control output based on the current measurement and time step.

        Args:
            measurement (float): The current value being measured.
            dt (float): The time step since the last measurement.

        Returns:
            float: The computed control output.
        """
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output
