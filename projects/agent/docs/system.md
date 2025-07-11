Agent System Overview
-----------------------------------------------------------------------------

                       +-------------------------+
                       |                         |
                       |    AgentCoordinator     |
                       |  (Main Orchestrator)    |
                       |                         |
                       +-------------------------+
                           /            |        \
                          /             |         \
          Control Flow   /              |          \   Control Flow
                        /               |           \
                       v                v            v
    +-------------------+    +------------------+    +------------------+
    |                   |    |                  |    |                  |
    | PerceptionProvider|    |   ModelWorker    |    | ActionExecutor   |
    | (Sensory Input)   |    | (Decision Making)|    | (Action Output)  |
    |                   |    |                  |    |                  |
    +-------------------+    +------------------+    +------------------+
              |                      ^                       ^
              |                      |                       |
              |     Data Flow        |      Data Flow        |
              +--------------------->+---------------------->+

Data/Control Flow Cycle:
1. PerceptionProvider → ModelWorker → ActionExecutor  (Data Flow)
2. AgentCoordinator manages and coordinates all components (Control Flow)

Component Functions:
--------------------
1. PerceptionProvider: Collects raw perceptions (sensor data, events) and makes them available 
    for processing.

2. ModelWorker: Processes perceptions using ML models to generate intelligent decisions
    based on current inputs and historical context.

3. ActionExecutor: Implements decisions by executing appropriate actions in the environment
    or target systems.

4. AgentCoordinator: Orchestrates the entire process by:
    - Managing perception acquisition from PerceptionProvider
    - Triggering ModelWorker for inference and decision-making
    - Directing ActionExecutor to implement decisions
    - Maintaining system timing and synchronization

AgentCoordinator and Execution Cycle (Per Tick):
---------------------------
1. Perceive: AgentCoordinator triggers PerceptionProvider to collect latest events
2. Think: AgentCoordinator activates ModelWorker to process perceptions and generate decisions
3. Act: AgentCoordinator instructs ActionExecutor to implement the decisions
4. Sleep: System waits until the next tick

**Important**: All these steps must complete within a single tick, with **non-blocking** manner.

Common Implementation Patterns:
------------------------------
1. Synchronous Model Inference:
    - Scenario: ML model inference is too heavy to run in parallel or system design requires sequential processing
    - Implementation:
    * ModelWorker should **abandons** the stale requests from the thinking queue regarding the inference time

2. Time-Suspended Inference:
    - Scenario: For simulation environments where model inference time should not affect simulation time
    - Implementation:
    * AgentCoordinator stops the simulation clock before queuing ModelWorker.
    * AgentCoordinator resumes the simulation clock after decision is made.
    * Even the desired design of AgentCoordinator is to be non-blocking, it must wait for the ModelWorker to finish in this case.
