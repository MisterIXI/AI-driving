import vgamepad as vg


def reset_gamepad(gamepad: vg.VX360Gamepad):
    gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gamepad.left_trigger_float(value_float=0.0)
    gamepad.right_trigger_float(value_float=0.0)
    gamepad.update()


class input_data:
    def __init__(self, steering: float, acceleration: float):
        self.joystick_input = steering
        if acceleration > 0:
            self.l_trigger = 0
            self.r_trigger = acceleration
        else:
            self.l_trigger = -acceleration
            self.r_trigger = 0

    def apply_to_gamepad(self, gamepad: vg.VX360Gamepad):
        gamepad.left_joystick_float(
            x_value_float=self.joystick_input, y_value_float=0.0)
        gamepad.left_trigger_float(value_float=self.l_trigger)
        gamepad.right_trigger_float(value_float=self.r_trigger)
        gamepad.update()
