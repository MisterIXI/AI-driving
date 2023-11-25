import threading
from inputs import get_gamepad

class tracker:
    def __init__(self):
        self.stick_x : float = 0.0
        self.RT : float = 0.0
        self.LT : float = 0.0
        self.A : bool = False
        self._thread : threading.Thread = None
        self.__active : bool = True
    
    def __poll_controller(self):
        print("Controller thread started")
        while self.__active:
            events = get_gamepad()
            for event in events:
                if event.code == "ABS_X":
                    self.stick_x = event.state / 32768.0
                elif event.code == "ABS_Z":
                    self.LT = event.state / 255.0
                elif event.code == "ABS_RZ":
                    self.RT = event.state / 255.0
                elif event.code == "BTN_SOUTH":
                    self.A = event.state == 1
        print("Controller thread exited")

    def start_tracking_agent(self):
        self.__active = True
        self._thread = threading.Thread(target=self.__poll_controller)
        self._thread.start()

    def stop_tracking_agent(self):
        self.__active = False
    
    def get_inputs(self) -> dict:
        return {
            "stick_x": self.stick_x,
            "RT": self.RT,
            "LT": self.LT,
            "A": self.A
        }
    
    def __str__(self):
        return f"stick_x: {self.stick_x}\nRT: {self.RT}\nLT: {self.LT}\nA: {self.A}"
        