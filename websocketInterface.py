from websocket import create_connection
import _thread
import time


class XBoardInterface():
    def __init__(self, name):
        self.ws = create_connection("ws://localhost:80/websocketclient")
        self.name = name
        self.gameStarted = False
        self.isMyTurn = False
        self.lastMove = None
        self.color = None

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
            self.isMyTurn = True
            if self.color == None:
                self.color = 'white'
        if message == "playother":
            self.gameStarted = True
            self.isMyTurn = False
            if self.color == None:
                self.color = 'black'
        if "move" in message and "pmove" not in message:
            self.lastMove = str(message)[-4:]
            self.isMyTurn = not self.isMyTurn
    
    def sendAction(self, message):
        # TODO remove 'move' correctly
        message = "move " + str(message)[-4:]
        print("[interface][" + self.name + "][action]:", message)
        self.ws.send(message)