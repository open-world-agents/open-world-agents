# agent/infrastructure/api.py
import logging
import threading
from typing import Any, Dict, Optional

from agent.application.task_executor import TaskManager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from shared.protocol import Message, MessageType, TaskResult, TaskSpecification

logger = logging.getLogger(__name__)

app = FastAPI(title="Desktop Agent API")
task_manager = TaskManager()


class MessageRequest(BaseModel):
    type: str
    payload: Dict[str, Any]


class MessageResponse(BaseModel):
    type: str
    payload: Dict[str, Any]


@app.post("/agent", response_model=MessageResponse)
async def process_message(request: MessageRequest):
    try:
        # Convert the request to our internal message format
        message = Message(type=MessageType(request.type), payload=request.payload)

        # Process the message based on its type
        if message.type == MessageType.PING:
            # Health check
            return MessageResponse(type=MessageType.READY.value, payload={"status": "ok"})

        elif message.type == MessageType.TASK:
            # Parse and start the task
            task_spec = TaskSpecification.from_dict(message.payload)
            success = task_manager.start_task(task_spec)

            if not success:
                return MessageResponse(
                    type=MessageType.ERROR.value,
                    payload={"error": "Could not start task", "details": "Another task is already running"},
                )

            return MessageResponse(type=MessageType.ACK.value, payload={"task_id": task_spec.id, "status": "started"})

        elif message.type == MessageType.STOP:
            # Stop the current task
            result = task_manager.stop_task()
            return MessageResponse(type=MessageType.COMPLETION.value, payload=result.to_dict())

        else:
            # Unknown message type
            return MessageResponse(
                type=MessageType.ERROR.value, payload={"error": f"Unknown message type: {message.type}"}
            )

    except Exception as e:
        logger.exception(f"Error processing message: {str(e)}")
        return MessageResponse(type=MessageType.ERROR.value, payload={"error": str(e)})


@app.get("/agent/status")
async def get_status():
    return task_manager.get_status()
