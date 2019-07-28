from websocket import create_connection
import sys
import _thread
import time


class XBoardInterface():
    def __init__(self, name, interfaceType, server_address):
        self.interfaceType = interfaceType
        if interfaceType == "websocket":
            self.ws = create_connection(server_address)
        self.name = name
        self.gameStarted = False
        self.isMyTurn = False
        self.lastMove = None
        self.otherMoves = []
        self.color = None

        # wait for messages
        _thread.start_new_thread(self._readWebsocket, ())

    def _readWebsocket(self):
        while True:
            if self.interfaceType == "websocket":
                result = str(self.ws.recv())
            else:
                result = str(sys.stdin.readline()).strip()
            self._handleServerMessage(result)

    def _handleServerMessage(self, message):
        self.logViaInterfaceType("[received]" + str(message))
        if message == "protover 4":
            self.sendViaInterfaceType("feature san=1, time=1, variants=\"bughouse\", myname==\"TandemTurtle\", otherboard=1, colors=1, time=1, done=1")

        if message == "go":
            self.gameStarted = True
            self.isMyTurn = True
            if self.color is None:
                self.color = 'white'
        if message == "playother":
            self.gameStarted = True
            self.isMyTurn = False
            if self.color is None:
                self.color = 'black'
        if "move" in message and "pmove" not in message and "Illegal" not in message:
            self.lastMove = self.stripMessage(message)
            self.isMyTurn = not self.isMyTurn
        if "pmove" in message:
            self.otherMoves += [self.stripMessage(message)]

    def sendAction(self, message):
        message = "move " + self.stripMessage(message)
        self.logViaInterfaceType("[action]:" + str(message))
        self.sendViaInterfaceType(message)

    def stripMessage(self, message):
        return str(message).split(' ')[-1]

    def logViaInterfaceType(self, message):
        if self.interfaceType == "websocket":
            print("[interface][" + self.name + "]" + message)

    def sendViaInterfaceType(self, message):
        if self.interfaceType == "websocket":
            self.ws.send(message)
        else:
            print(message)
