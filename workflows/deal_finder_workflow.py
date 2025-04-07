import asyncio
from temporalio import workflow
from temporalio.common import RetryPolicy
from datetime import timedelta
from typing import List, Dict, Any, TypedDict
import json

# Import activities
with workflow.unsafe.imports_passed_through():
    from activities.deal_finder_activities import DealFinderActivities

# --- Define Input/Output Types (based on TypeScript) ---

class RetrieveItemWorkflowRequest(TypedDict):
    llmEmbeddingModel: str
    llmModel: str
    query: str
    pineconeDBIndexes: List[str]

# Using Dict[str, Any] as placeholder for GroceryItemPineconeDBMetaSchema
GroceryItem = Dict[str, Any]

class StoreResult(TypedDict):
    results: List[GroceryItem]
    collection: str

AllStoreResultsWorkflowResponse = List[StoreResult]

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

# --- Helper Functions ---

def parse_price(price_str: Any) -> float:
    """Helper to safely parse price string into number, defaulting to Infinity if invalid."""
    if not isinstance(price_str, str):
        return float('inf')
    cleaned = ''.join(filter(lambda char: char.isdigit() or char == '.', price_str))
    try:
        parsed = float(cleaned)
        return parsed
    except ValueError:
        return float('inf')

def sort_items_by_price(items: List[GroceryItem]) -> List[GroceryItem]:
    """Sorting function (low to high), handles missing prices."""
    # Directly sort, using parse_price which handles missing/invalid prices by returning infinity
    # No longer filter out items missing the 'price' key beforehand.
    return sorted(items, key=lambda item: parse_price(item.get('price') if item else None))

# --- Workflow Definition ---

@workflow.defn
class DealFinderWorkflow:
    @workflow.run
    async def dealFinderItem(self, request: RetrieveItemWorkflowRequest) -> AllStoreResultsWorkflowResponse:
        workflow.logger.info(f"Starting DealFinderWorkflow for query: {request['query']}")

        llm_embedding_model = request['llmEmbeddingModel']
        llm_model = request['llmModel']
        query = request['query']
        # These are now treated as logical store names/filter tags
        store_filter_tags = request['pineconeDBIndexes']

        # 1. Get Query Embedding
        workflow.logger.info(f"Generating embedding for query: {query}")
        embedding_result = await workflow.execute_activity(
            DealFinderActivities.llm_embed,
            args=[llm_embedding_model, query],
            **activity_options
        )
        query_embedding = embedding_result['embeddings']

        # 2. Query the single index for each store filter in parallel
        workflow.logger.info(f"Querying single index in parallel with store filters: {store_filter_tags}")
        pinecone_query_tasks = []
        for store_tag in store_filter_tags:
            task = workflow.execute_activity(
                DealFinderActivities.pinecone_query,
                # Pass arguments positionally: embeddings, n_results (using default 10), store_tag
                args=[query_embedding, 10, store_tag],
                **activity_options
            )
            pinecone_query_tasks.append(task)

        # Gather results, handling potential errors for individual activities
        pinecone_results = await asyncio.gather(*pinecone_query_tasks, return_exceptions=True)

        query_results_by_store = {}
        for i, store_tag in enumerate(store_filter_tags):
            result = pinecone_results[i]
            if isinstance(result, Exception):
                workflow.logger.error(f"Failed to query with store filter {store_tag}: {result}")
                query_results_by_store[store_tag] = {"metadata": [], "possibleItems": []}
            else:
                # Store results keyed by the store tag
                possible_items = result.get('documents', [[]])[0]
                metadata = result.get('metadatas', [[]])[0]
                query_results_by_store[store_tag] = {
                    "metadata": metadata,
                    "possibleItems": possible_items
                }
                workflow.logger.info(f"Query successful for store filter: {store_tag}")


        # 3. Augment results and generate final output using LLM in parallel
        workflow.logger.info("Processing filtered results with LLM in parallel...")
        llm_tasks = []
        store_tags_for_llm = [] # Keep track of which tag corresponds to which task

        # Prepare LLM tasks for stores with results
        for store_tag, query_data in query_results_by_store.items():
            possible_items = query_data.get('possibleItems', [])
            metadata_list = query_data.get('metadata', []) # Keep metadata in case needed later, though not used in prompt

            if not possible_items:
                workflow.logger.warning(f"No possible items found from PineconeDB query for store filter {store_tag}. Skipping LLM step.")
                continue # Skip LLM if no items

            # Format the items clearly for the LLM
            formatted_items_list = "\n".join([f"- {item}" for item in possible_items])
            llm_prompt = (
                f"You are an AI assistant tasked with extracting relevant grocery items and their prices from a list based on a user query.\\n"
                f"User query: \"{query}\"\\n\\n"
                f"Below is a list of potential grocery items retrieved for this query. Each item string might contain the name, size/quantity, and price/deal information.\\n"
                f"{formatted_items_list}\\n\\n" # Use the clearly formatted list
                f"Instructions:\\n"
                f"1. Carefully review the 'User query'.\\n"
                f"2. From the list above, identify ONLY the items that are directly relevant to the 'User query'.\\n"
                f"3. For each relevant item, extract its primary name and its price (as a string, e.g., '1.99' or '5.49/lb'). \\n"
                f"   - Extract the price value found within the item string. If a price isn't clearly available for an item, use null for the price field.\\n"
                f"   - Extract just the core name of the item, omitting sizes or secondary details unless crucial for identification.\\n"
                f"4. Respond with a JSON array of objects. Each object must have two keys: 'name' (string) and 'price' (string or null).\\n"
                f"5. If NONE of the items in the list are relevant to the query, return an empty JSON array ([])."
            )

            # Define the expected JSON schema (same as before)
            llm_output_schema = {
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

            workflow.logger.info(f"Preparing LLM task for store filter: {store_tag}")
            task = workflow.execute_activity(
                DealFinderActivities.llm_generate,
                args=[llm_model, llm_prompt, "You are a helpful AI assistant. Respond ONLY with the requested JSON.", llm_output_schema],
                **activity_options
            )
            llm_tasks.append(task)
            store_tags_for_llm.append(store_tag) # Associate task with its store tag

        # Gather LLM results, handling potential errors
        final_results_per_index: AllStoreResultsWorkflowResponse = []
        if llm_tasks:
             llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)

             # Process LLM results
             for i, store_tag in enumerate(store_tags_for_llm):
                 result = llm_results[i]
                 if isinstance(result, Exception):
                     workflow.logger.error(f"Error processing LLM results for store filter {store_tag}: {result}")
                     final_results_per_index.append({
                         "results": [],
                         "collection": store_tag
                     })
                 else:
                     llm_extracted_items = result['response']
                     final_results_per_index.append({
                         "results": llm_extracted_items,
                         "collection": store_tag
                     })
                     workflow.logger.info(f"Successfully processed LLM results for store filter: {store_tag}")
        else:
            workflow.logger.info("No LLM tasks to execute.")

        workflow.logger.info("DealFinderWorkflow finished.")
        # Ensure the order matches the input `store_filter_tags` if necessary, or return as is.
        # Current implementation returns results in the order LLM tasks finished/failed.
        # If specific order is needed, sort `final_results_per_index` based on `store_filter_tags`.
        # For now, returning as is.
        return final_results_per_index