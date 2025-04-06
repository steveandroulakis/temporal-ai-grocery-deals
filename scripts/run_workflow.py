import asyncio
import uuid
import argparse  # Add argparse import
import sys  # Add sys import for argument checking
from temporalio.client import Client

# Import the workflow definition and its input type
from workflows.deal_finder_workflow import DealFinderWorkflow, RetrieveItemWorkflowRequest

from shared.config import get_temporal_client, TEMPORAL_TASK_QUEUE


async def main():
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Run the Deal Finder Temporal Workflow.")
    parser.add_argument(
        "--query",
        type=str,
        default="pasta",
        help="The search query for grocery items (default: 'pasta')."
    )
    args = parser.parse_args()

    # Check if default query is used and print message
    query_to_use = args.query
    if query_to_use == "pasta" and not any(arg.startswith('--query') for arg in sys.argv):
         print("No query provided via --query argument. Using default: 'pasta'")


    # Create a Temporal client
    client = await get_temporal_client()

    # --- Define the input for the DealFinderWorkflow ---
    # Use the query from arguments
    workflow_input = RetrieveItemWorkflowRequest(
        llmEmbeddingModel="text-embedding-3-small", # OpenAI embedding model
        llmModel="gpt-4o" , # OpenAI LLM
        query=query_to_use, # Use query from args
        # Example PineconeDB index names (e.g., store names) - Use exact names from data
        pineconeDBIndexes=["Safeway", "Trader Joe's", "Whole Foods"] # Changed store names
    )

    # Generate a unique ID for this workflow execution
    workflow_id = f"deal-finder-{uuid.uuid4()}"

    print(f"Starting DealFinderWorkflow with ID: {workflow_id}")
    print(f"Input Query: {workflow_input['query']}")
    print(f"Indexes (Store Tags): {workflow_input['pineconeDBIndexes']}") # Updated print label

    try:
        # Start the workflow execution
        handle = await client.start_workflow(
            DealFinderWorkflow.dealFinderItem, # The workflow run method
            workflow_input,                   # The input arguments
            id=workflow_id,
            task_queue=TEMPORAL_TASK_QUEUE,
        )

        print(f"Workflow started. Waiting for result...")

        # Wait for the workflow to complete and get the result
        result = await handle.result()

        print("\n--- Workflow Result ---")
        # Pretty print the result (assuming it's JSON-serializable like List[Dict])
        import json
        print(json.dumps(result, indent=2))
        print("---------------------")

    except Exception as e:
        print(f"\nError starting or executing workflow: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 