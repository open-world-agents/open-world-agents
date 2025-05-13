from owa.core import Runnable


class ModelWorker(Runnable):
    def on_configure(self, thought_queue, decision_queue, clock, model_id):
        self._thought_queue = thought_queue
        self._decision_queue = decision_queue
        self._clock = clock
        self._model_id = model_id

    def loop(self, *, stop_event):
        # Model inference logic
        pass
