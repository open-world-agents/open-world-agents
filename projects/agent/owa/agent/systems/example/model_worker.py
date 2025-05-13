import queue

from owa.agent.core import Clock
from owa.core import Runnable

from .processors import apply_processor


class ModelWorker(Runnable):
    def on_configure(self, thought_queue: queue.Queue[dict], decision_queue: queue.Queue, clock: Clock, model_id: str):
        self._thought_queue = thought_queue
        self._decision_queue = decision_queue
        self._clock = clock

        self._model, self._processor = None, None  # Placeholder for model and processor initialization

    def loop(self, *, stop_event):
        # Model inference logic
        # Placeholder for the actual model inference logic
        while not stop_event.is_set():
            thought = self._thought_queue.get()
            inputs = apply_processor(thought, processor=self._processor)
            # Process the thought and generate a decision
            decision = self.inference(inputs)
            self._decision_queue.put_nowait(decision)
            self._clock.sleep(1)

    def inference(self, inputs):
        # Placeholder for the actual processing logic
        # Implement your model inference logic here
        decision = f"No there's no {inputs['input_ids'][-len('dragon') :]}"
        return decision
