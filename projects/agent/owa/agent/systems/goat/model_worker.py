import queue

import torch
from transformers.models.smolvlm import SmolVLMForConditionalGeneration, SmolVLMProcessor

from owa.agent.core import Clock
from owa.core import Runnable

from .processors import apply_processor


def logits_processor(input_ids: torch.LongTensor, scores: torch.FloatTensor):
    """Process logits to adjust model generation behavior."""
    # Adjust token probabilities as needed
    # Example (commented out):
    # scores[:, 49354] *= 0.98  # <KEYBOARD_37_0> (left)
    # scores[:, 49358] *= 0.98  # <KEYBOARD_39_0> (right)
    return scores


class ModelWorker(Runnable):
    def on_configure(self, thought_queue: queue.Queue[dict], decision_queue: queue.Queue, clock: Clock, model_id: str):
        self._thought_queue = thought_queue
        self._decision_queue = decision_queue
        self._clock = clock

        self.model = SmolVLMForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
        self.processor = SmolVLMProcessor.from_pretrained(model_id)

    def loop(self, *, stop_event):
        # Model inference logic
        # Placeholder for the actual model inference logic
        while not stop_event.is_set():
            try:
                # NOTE: this sleep at least 1 seconds to avoid busy waiting
                thought = self._thought_queue.get(timeout=1)
            except queue.Empty:
                continue

            inputs = apply_processor(thought, processor=self.processor, is_training=False).to(
                self.model.device, dtype=self.model.dtype
            )
            # Process the thought and generate a decision
            decision = self.inference(inputs)
            self._decision_queue.put_nowait(decision)

    def inference(self, batch):
        # Placeholder for the actual processing logic
        # Implement your model inference logic here
        # decision = f"No there's no {inputs['input_ids'][-len('dragon') :]}"
        # decision = f"I want to press key {np.random.randint(1, 10)}"
        # self._clock.sleep(0.05)  # Simulate processing time

        outputs = self.model.generate(**batch, logits_processor=[logits_processor], do_sample=False, max_new_tokens=64)
        generated = self.processor.decode(outputs[0], skip_special_tokens=True)
        decision = generated[generated.find("Assistant: ") + len("Assistant: ") :]
        return decision
