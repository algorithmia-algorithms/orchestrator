from time import time

class Logger():
    def __init__(self):
        self.events = {}
        self.outcomes = {}

    def emit_event(self, namespace, message):
        event = {'message': message, "timestamp": str(time())}
        if namespace not in self.events:
            self.events[namespace] = []
            self.events[namespace].append(event)

    def emit_events(self, namespace, events):
        if namespace not in self.events:
            self.events[namespace] = []
        self.events[namespace] += events

    def emit_outcome(self, namespace, outcome):
        self.outcomes[namespace] = outcome

    def get_events(self):
        return self.events

    def get_outcomes(self):
        return self.outcomes