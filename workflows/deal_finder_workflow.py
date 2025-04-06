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
ACTIVITY_START_TO_CLOSE_TIMEOUT = timedelta(minutes=5)
ACTIVITY_SCHEDULE_TO_CLOSE_TIMEOUT = timedelta(minutes=6)
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

        # 2. Query the single index for each store filter
        workflow.logger.info(f"Querying single index with store filters: {store_filter_tags}")
        query_results_by_store = {}
        for store_tag in store_filter_tags:
            try:
                query_response = await workflow.execute_activity(
                    DealFinderActivities.pinecone_query,
                    # Pass arguments positionally: embeddings, n_results (using default 10), store_tag
                    args=[query_embedding, 10, store_tag],
                    **activity_options
                )
                # Store results keyed by the store tag
                possible_items = query_response.get('documents', [[]])[0]
                metadata = query_response.get('metadatas', [[]])[0]
                query_results_by_store[store_tag] = {
                    "metadata": metadata,
                    "possibleItems": possible_items
                }
                workflow.logger.info(f"Query successful for store filter: {store_tag}")
            except Exception as e:
                workflow.logger.error(f"Failed to query with store filter {store_tag}: {e}")
                # Store empty results for this tag on failure
                query_results_by_store[store_tag] = {"metadata": [], "possibleItems": []}
                continue

        # 3. Process results with LLM and refine
        workflow.logger.info("Processing filtered results with LLM...")
        final_results_per_index: AllStoreResultsWorkflowResponse = []

        # Iterate through the results we stored, keyed by store_tag
        for store_tag, query_data in query_results_by_store.items():
            possible_items = query_data.get('possibleItems', [])
            metadata_list = query_data.get('metadata', [])

            if not possible_items:
                workflow.logger.warning(f"No possible items found from PineconeDB query for store filter {store_tag}. Skipping LLM step.")
                final_results_per_index.append({
                    "results": [],
                    "collection": store_tag # Use store_tag as collection identifier
                })
                continue

            # Construct LLM prompt
            items_str = "\\n- ".join(possible_items)
            # Add the original query to the prompt for context
            llm_prompt = (
                f"You are an AI assistant filtering grocery items based on a user query.\n"
                f"User query: \"{query}\"\n\n" # Explicitly include the original user query
                f"Below is a list of potential grocery items (name and deal/price) retrieved for this query:\n"
                f"- {items_str}\n\n"
                f"Instructions:\n"
                f"1. Carefully review the 'User query'.\n"
                f"2. From the list above, select ONLY the strings that are directly relevant to the 'User query'.\n"
                f"3. Return the EXACT selected strings, without modification.\n"
                f"4. Respond with a JSON array containing ONLY the selected strings. \n"
                f"5. If NONE of the items in the list are relevant to the query, return an empty JSON array ([])."
            )

            try:
                workflow.logger.info(f"Calling LLM for store filter: {store_tag}")
                llm_response = await workflow.execute_activity(
                    DealFinderActivities.llm_generate,
                    # Pass system prompt and format hint positionally
                    args=[llm_model, llm_prompt, "You are a helpful AI assistant. Respond in JSON format.", {"type": "array", "items": {"type": "string"}}],
                    **activity_options
                )
                results_str = llm_response['response']
                filter_results: List[str]

                try:
                    filter_results = json.loads(results_str)
                    workflow.logger.info(f"LLM response parsed successfully for {store_tag}")
                except json.JSONDecodeError:
                    workflow.logger.warning(f"LLM response for {store_tag} was not valid JSON. Attempting repair...")
                    repaired_json = await workflow.execute_activity(
                        DealFinderActivities.json_repair,
                        args=[results_str],
                         **activity_options
                    )
                    try:
                         filter_results = json.loads(repaired_json)
                         workflow.logger.info(f"JSON repair successful for {store_tag}")
                    except json.JSONDecodeError:
                         workflow.logger.error(f"JSON repair failed for {store_tag}. Skipping results.")
                         filter_results = []

                # Remove duplicates
                filter_results = list(set(filter_results))

                # Refine results: Find metadata, handle misses with reverse search
                refined_items: List[GroceryItem] = []
                for item_str in filter_results:
                    # Direct find: Check if the metadata name is part of the LLM result string.
                    # This handles cases where LLM includes price/deal info in the string.
                    found_item = next((meta for meta in metadata_list 
                                       if isinstance(meta, dict) 
                                       and meta.get('name') 
                                       and meta['name'] in item_str), None)

                    if found_item:
                        # Ensure price exists in found metadata, parsing from item_str if necessary
                        if 'price' not in found_item or not found_item.get('price'):
                            try:
                                # Attempt to parse price from the end of item_str (e.g., "... (x.xx)")
                                price_part = item_str.split('(')[-1].split(')')[0]
                                found_item['price'] = str(parse_price(price_part)) # Store as string for consistency
                                workflow.logger.info(f"Parsed price '{found_item['price']}' from item_str '{item_str}' for direct match.")
                            except (IndexError, ValueError):
                                workflow.logger.warning(f"Could not parse price from item_str '{item_str}' for direct match.")
                                # Price remains missing or invalid, sort function will handle it.
                        
                        refined_items.append(found_item)
                    else:
                        workflow.logger.info(f"Item matching '{item_str}' not found directly in metadata for {store_tag}. Attempting reverse search (with filter).")
                        # Item Embedding
                        item_embedding_result = await workflow.execute_activity(
                            DealFinderActivities.llm_embed,
                            args=[llm_embedding_model, item_str],
                             **activity_options
                        )
                        item_embedding = item_embedding_result['embeddings']

                        # Reverse search - Apply the store filter tag positionally
                        item_query_response = await workflow.execute_activity(
                            DealFinderActivities.pinecone_query,
                            # Pass arguments positionally: item_embedding, n_results=1, store_tag
                            args=[item_embedding, 1, store_tag],
                            **activity_options
                        )

                        # Process reverse search results
                        reverse_found_meta_list = item_query_response.get('metadatas', [[[]]])[0]
                        if reverse_found_meta_list and isinstance(reverse_found_meta_list[0], dict):
                             reverse_found_item = reverse_found_meta_list[0]
                             if reverse_found_item.get('store') == store_tag:
                                 workflow.logger.info(f"Reverse search found item for '{item_str}' in {store_tag}")
                                 # Ensure price exists in found metadata, parsing from item_str if necessary
                                 if 'price' not in reverse_found_item or not reverse_found_item.get('price'):
                                      try:
                                          price_part = item_str.split('(')[-1].split(')')[0]
                                          reverse_found_item['price'] = str(parse_price(price_part))
                                          workflow.logger.info(f"Parsed price '{reverse_found_item['price']}' from item_str '{item_str}' for reverse match.")
                                      except (IndexError, ValueError):
                                          workflow.logger.warning(f"Could not parse price from item_str '{item_str}' for reverse match.")

                                 refined_items.append(reverse_found_item) # Append the found item
                             else:
                                 workflow.logger.warning(f"Reverse search for '{item_str}' found item, but store tag mismatch (expected {store_tag}, got {reverse_found_item.get('store')}). Skipping.")
                        else:
                            workflow.logger.warning(f"Reverse search failed for item '{item_str}' in {store_tag}")

                # Sort final items by price
                sorted_results = sort_items_by_price(refined_items)
                workflow.logger.info(f"Processed {len(sorted_results)} items for store filter {store_tag}")
                final_results_per_index.append({
                    "results": sorted_results,
                    "collection": store_tag # Use store_tag as collection identifier
                })

            except Exception as e:
                workflow.logger.error(f"Error processing LLM results for store filter {store_tag}: {e}")
                final_results_per_index.append({
                    "results": [],
                    "collection": store_tag # Use store_tag as collection identifier
                })

        workflow.logger.info("DealFinderWorkflow finished.")
        return final_results_per_index 