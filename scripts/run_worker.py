import asyncio
import concurrent.futures
import logging
from dotenv import load_dotenv

from temporalio.worker import Worker

# Import the Deal Finder workflow and activities
from workflows.deal_finder_workflow import DealFinderWorkflow
from activities.deal_finder_activities import DealFinderActivities

from shared.config import get_temporal_client, TEMPORAL_TASK_QUEUE


async def main():
    # Load environment variables
    load_dotenv(override=True)

    # --- Configure logging --- START
    # Basic configuration to show INFO level logs from activities
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # --- Configure logging --- END

    # Create the client
    client = await get_temporal_client()

    # Initialize the Deal Finder activities
    # Note: No LLM configuration needed here for now, as per request.
    deal_finder_activities = DealFinderActivities()
    logging.info("DealFinderActivities initialized.")

    # Prepare the list of activity methods from the instance
    activities_list = [
        deal_finder_activities.json_repair,
        deal_finder_activities.llm_embed,
        deal_finder_activities.llm_generate,
        deal_finder_activities.pinecone_query,
    ]

    logging.info("Worker ready to process DealFinder tasks!")

    # Run the worker
    # Using a ThreadPoolExecutor for activities. Adjust max_workers if needed.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as activity_executor:
        worker = Worker(
            client,
            task_queue=TEMPORAL_TASK_QUEUE,
            workflows=[DealFinderWorkflow],
            # Pass the list of activity methods
            activities=activities_list,
            activity_executor=activity_executor,
        )

        logging.info(f"Starting Deal Finder worker, connecting to task queue: {TEMPORAL_TASK_QUEUE}")
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main()) 