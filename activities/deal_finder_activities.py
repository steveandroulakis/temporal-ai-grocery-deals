import os
import yaml
import json
from temporalio import activity
from typing import List, Dict, Any, Optional
# Import Pinecone gRPC client components
from pinecone.grpc import PineconeGRPC as Pinecone, GRPCClientConfig
from pinecone import Index, QueryResponse
# Import OpenAI client
from openai import OpenAI, OpenAIError
# Import the new parsing utility
from helpers.parsing_utils import parse_llm_json_response
# Import the new host utility
from helpers.host_utils import determine_pinecone_host

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

    def __init__(self):
        """Initialize OpenAI and Pinecone clients once."""
        activity.logger.info("Initializing DealFinderActivities...")

        # --- Initialize OpenAI Client --- START
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            activity.logger.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.openai_client = OpenAI(api_key=openai_api_key)
        activity.logger.info("OpenAI client initialized.")

        # Determine embedding dimension
        try:
            dimension_str = os.environ.get("OPENAI_EMBEDDING_DIMENSION")
            self.embedding_dimension = int(dimension_str) if dimension_str else DEFAULT_OPENAI_EMBEDDING_DIMENSION
            activity.logger.info(f"Using embedding dimension: {self.embedding_dimension}")
        except ValueError:
            activity.logger.warning(f"Invalid OPENAI_EMBEDDING_DIMENSION value: {dimension_str}. Using default: {DEFAULT_OPENAI_EMBEDDING_DIMENSION}")
            self.embedding_dimension = DEFAULT_OPENAI_EMBEDDING_DIMENSION
        # --- Initialize OpenAI Client --- END


        # --- Initialize Pinecone Client --- START
        self.pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
        if not self.pinecone_index_name:
            activity.logger.error("PINECONE_INDEX_NAME environment variable not set.")
            raise ValueError("PINECONE_INDEX_NAME environment variable not set.")

        # Determine host using the helper function
        pinecone_grpc_host = determine_pinecone_host(
            index_name=self.pinecone_index_name,
            docker_compose_path=DOCKER_COMPOSE_PATH,
            logger=activity.logger
        )

        activity.logger.info(f"Connecting to Pinecone index '{self.pinecone_index_name}' via gRPC at {pinecone_grpc_host}...")
        self.pinecone_index: Optional[PineconeIndex] = None
        try:
            pc = Pinecone(api_key="dummy-key", host=pinecone_grpc_host, plaintext=True)
            self.pinecone_index = pc.Index(
                name=self.pinecone_index_name,
                host=pinecone_grpc_host,
                grpc_config=GRPCClientConfig(secure=False)
            )
            # Perform a dummy operation like describe_index_stats to check connection
            # index_stats = self.pinecone_index.describe_index_stats() # Optional check
            # activity.logger.info(f"Index Stats: {index_stats}")
            activity.logger.info(f"Successfully obtained gRPC index handle for '{self.pinecone_index_name}'.")

        except Exception as e:
            activity.logger.error(f"Error connecting to or getting index '{self.pinecone_index_name}' at {pinecone_grpc_host}: {e}")
            raise # Re-raise the exception

        if not self.pinecone_index or not hasattr(self.pinecone_index, 'query'):
             activity.logger.error("Failed to obtain a valid Pinecone index object.")
             raise TypeError("Failed to obtain a valid Pinecone index object for querying.")
        activity.logger.info("Pinecone client and index handle initialized.")
        # --- Initialize Pinecone Client --- END

        activity.logger.info("DealFinderActivities initialization complete.")


    @activity.defn
    async def llm_embed(self, model: str, input_text: str) -> Dict[str, List[float]]:
        """
        Activity using the pre-initialized OpenAI client to generate embeddings.
        Uses the 'model' parameter to specify the OpenAI embedding model.
        Returns the embedding vector itself under the 'embeddings' key.
        """
        activity.logger.info(f"llm_embed called for model '{model}' with input length: {len(input_text)}")
        # API Key check removed, client initialized in __init__
        # Dimension calculation removed, stored in self.embedding_dimension

        try:
            # Use the initialized client and dimension
            res = self.openai_client.embeddings.create(
                input=[input_text],
                model=model,
                dimensions=self.embedding_dimension # Use instance variable
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
        Activity using the pre-initialized OpenAI client to generate text.
        Supports system messages and JSON mode if format_hint={'type': 'json_object'} is provided.
        Uses the 'model' parameter to specify the OpenAI chat model.
        Returns the generated content under the 'response' key.
        """
        activity.logger.info(f"llm_generate called for model '{model}' with prompt length: {len(prompt)}")
        # API Key check removed, client initialized in __init__

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            # Use the initialized client
            # client = OpenAI(api_key=openai_api_key) # Removed

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

            completion = self.openai_client.chat.completions.create(**request_params) # Use initialized client

            response_content = completion.choices[0].message.content
            
            # --- NEW: Log the raw response very clearly --- START
            activity.logger.info("--- RAW LLM RESPONSE START ---")
            activity.logger.info(response_content)
            activity.logger.info("--- RAW LLM RESPONSE END ---")
            # --- NEW: Log the raw response very clearly --- END
            
            if response_content is None:
                 activity.logger.warning("OpenAI returned None content.")
                 response_content = "" # Default to empty string if OpenAI returns None
            else:
                # Minimal cleaning - just strip whitespace
                response_content = response_content.strip()
                activity.logger.info(f"Stripped response content length {len(response_content)} before parsing.")
                # NOTE: Removed complex JSON cleaning/parsing/re-serialization block.
                # The workflow is now responsible for parsing this string (which should be JSON).

            # --- Attempt to parse the LLM response string into a Python list using the utility --- START
            parsed_response = parse_llm_json_response(response_content, activity.logger)
            # --- Attempt to parse the LLM response string into a Python list using the utility --- END

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
        Queries the pre-initialized Pinecone index using a single embedding vector.
        Optionally filters results based on the 'store' metadata tag.
        Extracts 'original_chunk' for documents and full metadata.
        """
        # --- Determine Pinecone Host and Index Name --- REMOVED (Done in __init__)
        # --- Connect to Pinecone Index --- REMOVED (Done in __init__)

        # Use the pre-initialized index handle and name
        index = self.pinecone_index
        index_name = self.pinecone_index_name # Use instance variable

        # Check if index was successfully initialized
        if not index or not hasattr(index, 'query'):
             activity.logger.error("Pinecone index object is not valid or not initialized.")
             # Return empty results or raise an error, depending on desired behavior
             return {"documents": [[]], "metadatas": [[]]} # Example: return empty

        filter_dict = {"store": store_filter_tag} if store_filter_tag else None
        log_filter_msg = f"with filter: {filter_dict}" if filter_dict else "with no filter"
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