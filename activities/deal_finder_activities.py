import os
import yaml
import json
from temporalio import activity
from typing import List, Dict, Any, Optional
# Import Pinecone gRPC client components
from pinecone.grpc import PineconeGRPC as Pinecone, GRPCClientConfig
from pinecone import Index, QueryResponse
# Import OpenAI client
from openai import OpenAI, OpenAIError # Added OpenAIError

# Define placeholder types based on TypeScript usage
# Refined PineconeIndex type
PineconeIndex = Index
# This remains a Dict for now, as the structure can vary. Pydantic model is better for production.
GroceryItemPineconeDBMetaSchema = Dict[str, Any]

# Path to docker-compose relative to workspace root
DOCKER_COMPOSE_PATH = "pinecone/docker-compose.yaml"

# Default embedding dimension if not set in environment
# Should match the dimension used during data loading (e.g., 768 for text-embedding-3-small)
DEFAULT_OPENAI_EMBEDDING_DIMENSION = 768

class DealFinderActivities:

    @activity.defn
    async def json_repair(self, json_str: str) -> str:
        """
        Stub for the jsonRepair activity.
        Repairs a potentially invalid JSON string.
        """
        # In a real implementation, this would call a JSON repair library.
        # For now, it returns an empty JSON object string as a placeholder.
        activity.logger.info(f"Stub json_repair called with input length: {len(json_str)}")
        activity.logger.info("json_repair is currently a passthrough stub and not implemented.")
        return json_str

    @activity.defn
    async def llm_embed(self, model: str, input_text: str) -> Dict[str, List[float]]:
        """
        Activity using OpenAI to generate embeddings for the input text.
        Reads OPENAI_API_KEY and OPENAI_EMBEDDING_DIMENSION from environment.
        Uses the 'model' parameter to specify the OpenAI embedding model.
        Returns the embedding vector itself under the 'embeddings' key.
        """
        activity.logger.info(f"llm_embed called for model '{model}' with input length: {len(input_text)}")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            activity.logger.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        try:
            dimension_str = os.environ.get("OPENAI_EMBEDDING_DIMENSION")
            embedding_dimension = int(dimension_str) if dimension_str else DEFAULT_OPENAI_EMBEDDING_DIMENSION
            activity.logger.info(f"Using embedding dimension: {embedding_dimension}")
        except ValueError:
            activity.logger.warning(f"Invalid OPENAI_EMBEDDING_DIMENSION value: {dimension_str}. Using default: {DEFAULT_OPENAI_EMBEDDING_DIMENSION}")
            embedding_dimension = DEFAULT_OPENAI_EMBEDDING_DIMENSION

        try:
            client = OpenAI(api_key=openai_api_key)
            res = client.embeddings.create(
                input=[input_text],
                model=model,
                dimensions=embedding_dimension
            )
            embedding = res.data[0].embedding
            activity.logger.info(f"Successfully generated embedding with dimension {len(embedding)}")
            # Return the vector directly under the key expected by the workflow
            return {"embeddings": embedding}

        except OpenAIError as e:
            activity.logger.error(f"OpenAI API error during embedding generation: {e}")
            raise # Re-raise to fail the activity
        except Exception as e:
            activity.logger.error(f"Unexpected error during embedding generation: {e}")
            raise # Re-raise

    @activity.defn
    async def llm_generate(self, model: str, prompt: str, system: Optional[str] = None, format_hint: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Activity using OpenAI to generate text based on the prompt.
        Supports system messages and JSON mode if format_hint={'type': 'json_object'} is provided.
        Reads OPENAI_API_KEY from environment.
        Uses the 'model' parameter to specify the OpenAI chat model.
        Returns the generated content under the 'response' key.
        """
        activity.logger.info(f"llm_generate called for model '{model}' with prompt length: {len(prompt)}")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            activity.logger.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            client = OpenAI(api_key=openai_api_key)
            
            # --- Determine OpenAI request parameters --- START
            request_params = {
                "model": model,
                "messages": messages,
            }
            # Explicitly request JSON output if hinted
            if format_hint and format_hint.get("type") in ["json_object", "array"]: # Check for 'array' too
                 activity.logger.info("Requesting JSON object format from OpenAI.")
                 request_params["response_format"] = {"type": "json_object"}
            # --- Determine OpenAI request parameters --- END

            completion = client.chat.completions.create(**request_params) # Use prepared params

            response_content = completion.choices[0].message.content
            
            # --- NEW: Log the raw response very clearly --- START
            activity.logger.info("--- RAW LLM RESPONSE START ---")
            activity.logger.info(response_content)
            activity.logger.info("--- RAW LLM RESPONSE END ---")
            # --- NEW: Log the raw response very clearly --- END
            
            # --- Debugging: Log raw response content --- START
            activity.logger.info(f"Raw OpenAI response content: {response_content}")
            # --- Debugging: Log raw response content --- END
            
            if response_content is None:
                 activity.logger.warning("OpenAI returned None content.")
                 response_content = "[]" # Default to empty JSON array string if OpenAI returns None
            else:
                # Minimal cleaning - just strip whitespace
                response_content = response_content.strip()
                activity.logger.info(f"Returning stripped response content with length {len(response_content)}")
                # NOTE: Removed complex JSON cleaning/parsing/re-serialization block.
                # The workflow is now responsible for parsing this string (which should be JSON).

            # --- Attempt to parse the LLM response string into a Python list --- START
            parsed_response: List[Dict[str, Any]] = [] # Default to empty list
            try:
                parsed_obj = json.loads(response_content)

                if isinstance(parsed_obj, list):
                    # Handles [...] and []
                    # Check if all items are dicts and have the required keys
                    if all(isinstance(item, dict) and 'name' in item and 'price' in item for item in parsed_obj):
                        parsed_response = parsed_obj
                        activity.logger.info(f"Successfully parsed LLM response into a list of {len(parsed_response)} dicts.")
                    else:
                        activity.logger.warning("Parsed JSON list did not contain only dictionaries with 'name' and 'price' keys. Returning empty list.")
                        parsed_response = [] # Keep strict: require name/price

                elif isinstance(parsed_obj, dict):
                    # Handles the case where LLM returns just one item as an object {}
                    if 'name' in parsed_obj and 'price' in parsed_obj:
                        activity.logger.info("Parsed LLM response as a single dictionary, wrapping it in a list.")
                        parsed_response = [parsed_obj]
                    else:
                        # NEW: Check if dict contains a key whose value is the desired list of dicts
                        found_list_in_dict = False
                        for key, value in parsed_obj.items():
                            # Check if the value is a list, and all its items are dicts with 'name' and 'price'
                            if isinstance(value, list) and all(isinstance(item, dict) and 'name' in item and 'price' in item for item in value):
                                activity.logger.info(f"Found expected list of dicts under key '{key}' in the response object.")
                                parsed_response = value
                                found_list_in_dict = True
                                break # Use the first suitable list found
                        
                        if not found_list_in_dict:
                           activity.logger.warning("Parsed dictionary object was not a single item with name/price and did not contain a recognized nested list of items. Returning empty list.")
                           parsed_response = []

                # Commenting out the very specific nested string case for simplicity unless proven necessary
                # elif isinstance(parsed_obj, list) and len(parsed_obj) == 1 and isinstance(parsed_obj[0], str):
                #    ... (removed nested parsing logic) ...

                else:
                    # If it's not a list or a dict we can handle, log warning and return empty
                    activity.logger.warning(f"LLM JSON response was not a list or a recognized dictionary structure (Type: {type(parsed_obj)}). Returning empty list.")
                    parsed_response = [] # Explicitly set default

            except json.JSONDecodeError as json_err:
                activity.logger.error(f"Failed to parse LLM response string as JSON: {json_err}. Response: {response_content}")
                parsed_response = [] # Ensure [] on JSON error
            except Exception as e:
                 activity.logger.error(f"Unexpected error during final response parsing: {e}. Returning empty list.")
                 parsed_response = [] # Ensure [] on other errors
            # --- Attempt to parse the LLM response string into a Python list --- END

            # Return the parsed Python list (or empty list on failure)
            return {"response": parsed_response}

        except OpenAIError as e:
            activity.logger.error(f"OpenAI API error during chat completion: {e}")
            raise # Re-raise to fail the activity
        except Exception as e:
            activity.logger.error(f"Unexpected error during chat completion: {e}")
            raise # Re-raise

    @activity.defn
    async def pinecone_query(self, query_embedding: List[float], n_results: int = 10, store_filter_tag: Optional[str] = None) -> Dict[str, List[List[Any]]]:
        """
        Connects to the single Pinecone index (using env PINECONE_INDEX_NAME
        and deriving port from docker-compose.yaml), then queries it using
        a single embedding vector via the gRPC client.
        Optionally filters results based on the 'store' metadata tag.
        Extracts 'original_chunk' for documents and full metadata.
        """
        # --- Determine Pinecone Host and Index Name --- START
        env_index_name = os.environ.get("PINECONE_INDEX_NAME")
        if not env_index_name:
            activity.logger.error("PINECONE_INDEX_NAME environment variable not set.")
            raise ValueError("PINECONE_INDEX_NAME environment variable not set.")

        # Find docker-compose path relative to likely workspace root
        # Note: This assumes the worker runs with the workspace root as CWD
        # or that the relative path from CWD to workspace root is consistent.
        # A more robust solution might involve setting WORKSPACE_ROOT env var.
        docker_compose_abs_path = os.path.abspath(DOCKER_COMPOSE_PATH)
        if not os.path.exists(docker_compose_abs_path):
             # Try path relative to this script's dir if not found from CWD
             script_dir = os.path.dirname(__file__)
             rel_path = os.path.join(script_dir, '..', DOCKER_COMPOSE_PATH)
             docker_compose_abs_path = os.path.abspath(rel_path)
             if not os.path.exists(docker_compose_abs_path):
                  activity.logger.error(f"docker-compose.yaml not found at {DOCKER_COMPOSE_PATH} or relative path. Cannot determine port.")
                  raise FileNotFoundError("docker-compose.yaml not found, cannot determine Pinecone port.")

        # Load docker-compose and extract port
        pinecone_grpc_host = None
        try:
            with open(docker_compose_abs_path, 'r') as f:
                config = yaml.safe_load(f)
            if not config or 'services' not in config or env_index_name not in config['services']:
                raise ValueError(f"Invalid docker-compose file or service '{env_index_name}' not found.")
            service_config = config['services'][env_index_name]
            if 'ports' in service_config and service_config['ports']:
                port_mapping = service_config['ports'][0]
                port = int(port_mapping.split(':')[0]) # Get the host port
                pinecone_grpc_host = f"localhost:{port}"
                activity.logger.info(f"Determined Pinecone gRPC host from docker-compose: {pinecone_grpc_host}")
            else:
                raise ValueError(f"No port mapping found for service '{env_index_name}'.")
        except (yaml.YAMLError, ValueError, IndexError, TypeError, FileNotFoundError) as e:
            activity.logger.error(f"Error reading docker-compose or extracting port: {e}")
            raise # Re-raise to fail the activity

        if not pinecone_grpc_host:
             # This case should be caught by exceptions above, but as a safeguard:
             activity.logger.error("Failed to determine pinecone_grpc_host.")
             raise ValueError("Failed to determine pinecone_grpc_host.")

        # --- Determine Pinecone Host and Index Name --- END

        # --- Connect to Pinecone Index --- START
        activity.logger.info(f"Connecting to Pinecone index '{env_index_name}' via gRPC at {pinecone_grpc_host} within pinecone_query activity...")
        index: Optional[PineconeIndex] = None
        try:
            # Use the dynamically determined host
            pc = Pinecone(api_key="dummy-key", host=pinecone_grpc_host, plaintext=True)
            index = pc.Index(
                name=env_index_name,
                host=pinecone_grpc_host,
                grpc_config=GRPCClientConfig(secure=False)
            )
            activity.logger.info(f"Successfully obtained gRPC index handle for '{env_index_name}'.")
        except Exception as e:
            activity.logger.error(f"Error connecting to or getting index '{env_index_name}' at {pinecone_grpc_host}: {e}")
            raise # Re-raise the exception to fail the activity

        if not index or not hasattr(index, 'query'):
             activity.logger.error("Failed to obtain a valid Pinecone index object.")
             raise TypeError("Failed to obtain a valid Pinecone index object for querying.")
        # --- Connect to Pinecone Index --- END

        # Use the connected index name for logging
        index_name = env_index_name # Keep for logging consistency
        filter_dict = {"store": store_filter_tag} if store_filter_tag else None # Re-enabled filter
        # filter_dict = None # <--- TEMPORARILY DISABLED FILTER FOR DEBUGGING
        log_filter_msg = f"with filter: {filter_dict}" if filter_dict else "with no filter" # Adjusted log msg
        # Updated log message for single embedding
        activity.logger.info(f"Querying index '{index_name}' with 1 embedding, top_k={n_results} {log_filter_msg}")

        # Validate the received embedding
        if not query_embedding:
            activity.logger.warn(f"Received empty or invalid query embedding for index '{index_name}'. Returning empty results.")
            return {"documents": [[]], "metadatas": [[]]}

        # Use the received embedding directly
        query_vector = query_embedding

        # --- Debugging: Log type and sample of the vector --- START
        try:
            activity.logger.info(f"Type of query_vector: {type(query_vector)}")
            if isinstance(query_vector, list):
                activity.logger.info(f"Length of query_vector: {len(query_vector)}")
                # Log first 5 elements if available
                sample_elements = query_vector[:5]
                activity.logger.info(f"Sample elements of query_vector: {sample_elements}")
            else:
                activity.logger.warning("query_vector is not a list as expected!")
        except Exception as log_e:
            activity.logger.error(f"Error during debug logging: {log_e}")
        # --- Debugging: Log type and sample of the vector --- END

        try:
            query_response: QueryResponse = index.query(
                vector=query_vector,
                top_k=n_results,
                filter=filter_dict, # Apply the filter
                include_metadata=True,
                include_values=False # Values (embeddings) usually not needed in response
            )

            documents = []
            metadatas = []
            if query_response.matches:
                activity.logger.info(f"Received {len(query_response.matches)} matches from Pinecone query on '{index_name}'.")
                for match in query_response.matches:
                    # Ensure metadata exists and is a dictionary
                    metadata = match.metadata if isinstance(match.metadata, dict) else {}
                    # Extract the 'original_chunk' text, default to empty string if not found
                    documents.append(metadata.get('original_chunk', ''))
                    # Append the full metadata dictionary
                    metadatas.append(metadata)
            else:
                 activity.logger.info(f"No matches found from Pinecone query on '{index_name}'.")


            # Return in the nested list format expected by the workflow
            return {
                "documents": [documents],
                "metadatas": [metadatas]
            }
        except Exception as e:
            activity.logger.error(f"Error querying Pinecone index '{index_name}': {e}")
            # Return empty results on error to allow workflow to potentially continue
            return {"documents": [[]], "metadatas": [[]]}

# Note: The type GroceryItemPineconeDBMetaSchema is a placeholder.
# For production, define a specific Pydantic model for metadata structure. 