from datasets import load_dataset
from trl import SFTTrainer
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any
import uvicorn
import asyncio
import time
import logging
import os

app = FastAPI()

logger = logging.getLogger("uvicorn")  # Get the uvicorn logger
logger.info("Starting the app")


def _exit_process(exit_code: int) -> None:
    # This function is called to exit the process after the response is sent, to finish the job.
    time.sleep(0.1)  # Small delay to ensure response has flushed
    os._exit(
        exit_code
    )  # Exit the process with the exit code, use os._exit instead of sys.exit to suppress ASGI application exception


async def dataset():
    logger.info("Loading dataset ...")
    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")
    return dataset


async def train(dataset):
    logger.info("Running training job")
    trainer = SFTTrainer(model="Qwen/Qwen2-0.5B-Instruct", train_dataset=dataset)
    trainer.train()


@app.post("/job")
async def main():
    await train(await dataset())
    BackgroundTasks.add_task(_exit_process, 1)


# Health check endpoint is required for batch jobs
@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, log_level="info")
