import Algorithmia
import numpy as np
from threading import Thread
from time import time

ALGO_A = "algo://network_anomaly_detection/algo_a/0.2.x"
ALGO_B = "algo://network_anomaly_detection/algo_b/0.1.x"

client = Algorithmia.client()


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


def call_algo_a(input: np.ndarray, logger: Logger):
    formatted_input = input.tolist()
    logger.emit_event("HOST", "starting algorithm A")
    algo_response = client.algo(ALGO_A).pipe(formatted_input)
    result = algo_response.result
    logger.emit_event("HOST", "completed algorithm A")
    logger.emit_events("ALGO_A", result['events'])
    logger.emit_outcome("ALGO_A", result['outcome'])
    return logger


def call_algo_b(input: np.ndarray, logger: Logger):
    formatted_input = input.tolist()
    logger.emit_event("HOST", "starting algorithm B")
    algo_response = client.algo(ALGO_B).pipe(formatted_input)
    result = algo_response.result
    logger.emit_event("HOST", "completed algorithm B")
    logger.emit_events("ALGO_B", result['events'])
    logger.emit_outcome("ALGO_B", result['outcome'])
    return logger


def apply(input):
    logger = Logger()
    logger.emit_event("HOST", "algorithm workflow started")
    try:
        if isinstance(input, dict):
            if "features" in input and isinstance(input['features'], list):
                input_tensor = np.asarray(input['features'])
                logger.emit_event('HOST', "input tensor converted successfully")
            else:
                raise Exception("'features' was either not defined, or was an invalid type.")
            if "device_type" in input and isinstance(input['device_type'], str):
                if input['device_type'] == "DEVICE_A":
                    logger.emit_event('HOST', "'device_type' was 'DEVICE_A'")
                    model_a_data = input_tensor[:, 0:5]
                    model_b_data = input_tensor[:, 5:]
                elif input['device_type'] == "DEVICE_B":
                    logger.emit_event('HOST', "'device_type' was 'DEVICE_B'")
                    model_a_data = input_tensor[:, 1:6]
                    model_b_data = input_tensor[:, 4:-1]
                else:
                    raise Exception("'device_type must be 'DEVICE_A' or 'DEVICE_B'")
            else:
                raise Exception("'device_type' was either not defined, or was an invalid type")

        else:
            raise Exception("input is of type {}, must be a json object.".format(type(input)))

        threads = [Thread(target=call_algo_a, args=(model_a_data, logger)),
                   Thread(target=call_algo_b, args=(model_b_data, logger))]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
        output = {"events": logger.get_events(), "outcomes": logger.get_outcomes()}
        return output
    except Exception as e:
        logger.emit_event("HOST", str(e))
        raise e

if __name__ == "__main__":
    input = {"features": [[4.0, 3.0, 2.2, -1.1, 0.25, 0.15, -0.241, 0.05, 0.99, -0.25]], "device_type": "DEVICE_B"}
    result = apply(input)
    print(result)