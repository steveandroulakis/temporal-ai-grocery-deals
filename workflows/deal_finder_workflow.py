import asyncio
from temporalio import workflow
from temporalio.common import RetryPolicy
from datetime import timedelta
from typing import List, Dict, Any, TypedDict, Optional

# Import activities
with workflow.unsafe.imports_passed_through():
    from activities.deal_finder_activities import DealFinderActivities

# --- Type Definitions (using snake_case) ---

class RetrieveItemWorkflowRequest(TypedDict):
    llm_embedding_model: str
    llm_model: str
    query: str
    store_filter_tags: List[str] # Renamed from pineconeDBIndexes

class GroceryItem(TypedDict):
    name: str
    price: Optional[str] # Price can be string or null

class StoreResult(TypedDict):
    results: List[GroceryItem]
    collection: str # Represents the store_tag
    error: Optional[bool] # Optional flag to indicate processing errors

# Final workflow response type
AllStoreResultsWorkflowResponse = List[StoreResult]

# --- LLM Prompting Constants ---

LLM_SYSTEM_PROMPT = "You are a helpful AI assistant. Respond ONLY with the requested JSON."

LLM_USER_PROMPT_TEMPLATE = (
    "You are an AI assistant tasked with extracting relevant grocery items and their prices from a list based on a user query.\n"
    "User query: \"{query}\"\n\n"
    "Below is a list of potential grocery items retrieved for this query. Each item string might contain the name, size/quantity, and price/deal information.\n"
    "{formatted_items_list}\n\n"
    "Instructions:\n"
    "1. Carefully review the 'User query'.\n"
    "2. From the list above, identify ONLY the items that are directly relevant to the 'User query'.\n"
    "3. For each relevant item, extract its primary name and its price (as a string, e.g., '1.99' or '5.49/lb'). \n"
    "   - Extract the price value found within the item string. If a price isn't clearly available for an item, use null for the price field.\n"
    "   - Extract just the core name of the item, omitting sizes or secondary details unless crucial for identification.\n"
    "4. Respond with a JSON array of objects. Each object must have two keys: 'name' (string) and 'price' (string or null).\n"
    "5. If NONE of the items in the list are relevant to the query, return an empty JSON array ([])."
)

LLM_OUTPUT_SCHEMA_HINT = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": ["string", "null"]}
        },
        "required": ["name", "price"]
    }
}

# --- Temporal Activity Configuration ---

# Use timeouts similar to agent_goal_workflow or adjust as needed
ACTIVITY_START_TO_CLOSE_TIMEOUT = timedelta(seconds=12)
ACTIVITY_SCHEDULE_TO_CLOSE_TIMEOUT = timedelta(minutes=1)
RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=1.5, # Slightly more aggressive backoff than agent_goal
    maximum_attempts=3,
)

# Helper dictionary for common activity options
activity_options = {
    "start_to_close_timeout": ACTIVITY_START_TO_CLOSE_TIMEOUT,
    "schedule_to_close_timeout": ACTIVITY_SCHEDULE_TO_CLOSE_TIMEOUT,
    "retry_policy": RETRY_POLICY,
}

@workflow.defn
class DealFinderWorkflow:
    @workflow.run
    async def dealFinderItem(self, request: RetrieveItemWorkflowRequest) -> AllStoreResultsWorkflowResponse:
        workflow.logger.info(f"Starting DealFinderWorkflow for query: {request['query']}")

        # Extract request parameters using snake_case keys
        llm_embedding_model = request['llm_embedding_model']
        llm_model = request['llm_model']
        query = request['query']
        store_filter_tags = request['store_filter_tags']

        # 1. Get Query Embedding
        workflow.logger.info(f"Generating embedding for query: '{query}'")
        try:
            embedding_result = await workflow.execute_activity(
                DealFinderActivities.llm_embed,
                args=[llm_embedding_model, query],
                **activity_options
            )
            query_embedding = embedding_result.get('embeddings')
            if not query_embedding:
                 workflow.logger.error("Failed to get embeddings from llm_embed activity result.")
                 # Decide how to handle this: fail workflow or return empty for all stores?
                 # Failing fast here:
                 raise ValueError("Failed to generate query embedding.")
        except Exception as e:
             workflow.logger.error(f"Error generating query embedding: {e}")
             raise # Re-raise to fail the workflow

        # --- Inner function to process each store concurrently ---
        async def process_single_store(store_tag: str, query_embedding: List[float], llm_model: str, query: str) -> StoreResult:
            """Queries Pinecone and calls LLM for a single store filter tag."""
            workflow.logger.info(f"Processing store filter: {store_tag}")
            try:
                # 1. Pinecone Query for this store
                pinecone_result = await workflow.execute_activity(
                    DealFinderActivities.pinecone_query,
                    args=[query_embedding, 10, store_tag], # Get top 10 results
                    **activity_options
                )

                # Safely extract documents and metadata
                # The activity returns [[docs]], [[metadatas]] structure
                documents_list = pinecone_result.get('documents')
                possible_items = documents_list[0] if documents_list and isinstance(documents_list, list) and len(documents_list) > 0 else []
                # metadatas_list = pinecone_result.get('metadatas') # Metadata not currently used downstream, can uncomment if needed
                # metadata = metadatas_list[0] if metadatas_list and isinstance(metadatas_list, list) and len(metadatas_list) > 0 else []

                workflow.logger.info(f"Store '{store_tag}': Found {len(possible_items)} potential items from Pinecone.")

                # 2. LLM Generation (if items found)
                if possible_items:
                    formatted_items_list = "\n".join([f"- {item}" for item in possible_items])
                    llm_prompt = LLM_USER_PROMPT_TEMPLATE.format(
                        query=query,
                        formatted_items_list=formatted_items_list
                    )

                    workflow.logger.info(f"Store '{store_tag}': Calling LLM to extract relevant items.")
                    llm_result = await workflow.execute_activity(
                        DealFinderActivities.llm_generate,
                        args=[llm_model, llm_prompt, LLM_SYSTEM_PROMPT, LLM_OUTPUT_SCHEMA_HINT],
                        **activity_options
                    )
                    # The activity now returns the parsed list directly
                    extracted_items: List[GroceryItem] = llm_result.get('response', [])
                    workflow.logger.info(f"Store '{store_tag}': LLM extracted {len(extracted_items)} items.")
                    return {"results": extracted_items, "collection": store_tag}
                else:
                    # No items found in Pinecone for this store
                    workflow.logger.info(f"Store '{store_tag}': No items found, skipping LLM call.")
                    return {"results": [], "collection": store_tag}

            except Exception as e:
                # Log the error and return an empty result for this store, marking error
                workflow.logger.error(f"Error processing store '{store_tag}': {e}")
                # Indicate error for potential upstream handling
                return {"results": [], "collection": store_tag, "error": True}
        # --- End of inner function ---

        # --- Gather results from all stores concurrently ---
        workflow.logger.info(f"Starting concurrent processing for stores: {store_filter_tags}")
        store_processing_tasks = [
            process_single_store(tag, query_embedding, llm_model, query)
            for tag in store_filter_tags
        ]

        final_results_per_store = await asyncio.gather(*store_processing_tasks)
        workflow.logger.info("Finished processing all stores.")
        return final_results_per_store # Type: AllStoreResultsWorkflowResponse