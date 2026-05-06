# TODO: propose oep for evaluator-agent design.

# ================ Definition of the `agent` API ================================

from owa.evaluate import Evaluator


def reset_agent(data): ...  # user-defined function
def stop_agent(): ...  # user-defined function
def close_agent(): ...  # user-defined function


evaluator = Evaluator(reset_fn=reset_agent, stop_fn=stop_agent, close_fn=close_agent)
eval_result = evaluator.evaluate()  # takes some time to run
print(eval_result)


# ================ Example of the implementation of the `agent` backend ================================
# Invididual ML Researcher writes the following code. Only example is provided in repository.

from fastapi import FastAPI

app = FastAPI()


# Define API endpoints
@app.post("/reset")
def reset_agent(data): ...


@app.get("/stop")
def stop_agent(): ...


@app.get("/close")
def close_agent(): ...
