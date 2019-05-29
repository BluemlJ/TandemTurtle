from websocket import create_connection
import _thread
import time


class XBoardInterface():
    def __init__(self, name):
        self.ws = create_connection("ws://localhost:80/websocketclient")
        self.name = name
        self.gameStarted = False

        # wait for messages
        _thread.start_new_thread( self._readWebsocket, ())
  
    def _readWebsocket(self):
        count = 0
        while True:
            result = str(self.ws.recv())
            self._handleServerMessage(result)

    def _handleServerMessage(self, message):
        print("[interface][" + self.name + "][received]", message)
        if message == "protover 4":
            self.ws.send("feature san=1, time=1, variants=\"bughouse\", otherboard=1, colors=1, time=1, done=1")
        
        if message == "go":
            self.gameStarted = True

    
    def sendAction(self, message):
        message = str(message)
        print("[interface][" + self.name + "][action]:", message)
        self.ws.send(message)