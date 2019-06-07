from websocket import create_connection
import time


class XBoardInterface():
    def __init__(self, name):
        self.name = name
        self.gameStarted = True

    def sendAction(self, message):
        message = str(message)
        print("[interface][" + self.name + "][action]:", message)
