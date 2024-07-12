import vgamepad as vg


class gamepad:
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
        self.joystick_input = 0.0
        self.l_trigger = 0.0
        self.r_trigger = 0.0

    def reset(self):
        self.joystick_input = 0.0
        self.l_trigger = 0.0
        self.r_trigger = 0.0
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.left_trigger_float(value_float=0.0)
        self.gamepad.right_trigger_float(value_float=0.0)
        self.gamepad.update()

    def apply_input(self, steering: float, acceleration: float):
        self.joystick_input = steering
        print(acceleration)
        if acceleration > 0:
            self.l_trigger = 0.0
            self.r_trigger = acceleration
        else:
            self.l_trigger = -acceleration
            self.r_trigger = 0.0
        self.gamepad.left_joystick_float(
            x_value_float=self.joystick_input, y_value_float=0.0)
        self.gamepad.left_trigger_float(value_float=self.l_trigger)
        self.gamepad.right_trigger_float(value_float=self.r_trigger)
        self.gamepad.update()

    # def get_input(self):
    #     return input_data(self.joystick_input, self.r_trigger-self.l_trigger)
    def print_input(self):
        print(str(self))
        steer = "steering:     ["
        accel = "acceleration: ["
        # get steer of 21 bins
        steer_i = int((self.joystick_input+1)*10)
        accel_i = int(((self.r_trigger-self.l_trigger)+1)*10)
        for i in range(21):
            if i == steer_i:
                steer += "+"
            elif i == 11:
                steer += "|"
            else:
                steer += " "
            if i == accel_i:
                accel += "+"
            elif i == 11:
                accel += "|"
            else:
                accel += " "
        steer += "]" + str(self.joystick_input)
        accel += "]" + str(self.r_trigger-self.l_trigger)
        print(steer)
        print(accel)

    def __str__(self) -> str:
        return f"gamepad: {self.joystick_input}, {self.l_trigger}, {self.r_trigger}"


if __name__ == "__main__":
    import time
    gp = gamepad()
    while True:
        gp.apply_input(1.0, 1.0)
        gp.print_input()
        time.sleep(1)
        gp.apply_input(-1.0, -1.0)
        gp.print_input()
        time.sleep(1)
        gp.reset()
        time.sleep(1)