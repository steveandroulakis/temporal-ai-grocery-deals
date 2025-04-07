import os
import json
from typing import List, Dict, Any, Optional

# Third-party imports
from temporalio import activity
from openai import OpenAI, OpenAIError
from pinecone import Index as PineconeIndex # Type alias
from pinecone import QueryResponse # Type alias for clarity in pinecone_query

# Local application imports
from helpers.parsing_utils import parse_llm_json_response
# Import the new helper functions
from helpers.config_utils import get_env_var, get_embedding_dimension
from helpers.host_utils import init_pinecone_index

# Define placeholder types for clarity (consider Pydantic for production)
# PineconeIndex defined above via import alias
GroceryItemPineconeDBMetaSchema = Dict[str, Any]

# Constants (Keep dimension default here as it's tied to activities logic)
DEFAULT_OPENAI_EMBEDDING_DIMENSION = 768  # e.g., for text-embedding-3-small

class DealFinderActivities:
    """
    Contains activities for finding grocery deals, interacting with OpenAI and Pinecone.
    Clients are initialized once upon worker startup using helper functions.
    """

    def __init__(self):
        """Initializes OpenAI and Pinecone clients using helper functions."""
        activity.logger.info("Initializing DealFinderActivities...")
        log = activity.logger # Use a local alias for brevity

        # Load configurations using helpers
        self.openai_api_key = get_env_var("OPENAI_API_KEY", log)
        self.pinecone_index_name = get_env_var("PINECONE_INDEX_NAME", log)
        self.embedding_dimension = get_embedding_dimension(log)

        # Initialize OpenAI client
        try:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            log.info("OpenAI client initialized.")
        except Exception as e:
            log.error(f"Failed to initialize OpenAI client: {e}")
            raise

        # Initialize Pinecone client using helper
        self.pinecone_index = init_pinecone_index(self.pinecone_index_name, log)

        log.info("DealFinderActivities initialization complete.")

    @activity.defn
    async def llm_embed(self, model: str, input_text: str) -> Dict[str, List[float]]:
        """
        Generates embeddings for the input text using the specified OpenAI model.

        Args:
            model: The OpenAI embedding model to use (e.g., "text-embedding-3-small").
            input_text: The text to generate embeddings for.

        Returns:
            A dictionary containing the embedding vector under the 'embeddings' key.
        """
        activity.logger.info(f"Generating embedding with model '{model}' for input length: {len(input_text)}")

        try:
            res = self.openai_client.embeddings.create(
                input=[input_text],
                model=model,
                dimensions=self.embedding_dimension
            )
            embedding = res.data[0].embedding
            activity.logger.info(f"Successfully generated embedding with dimension {len(embedding)}")
            # Return structure expected by the workflow
            return {"embeddings": embedding}

        except OpenAIError as e:
            activity.logger.error(f"OpenAI API error during embedding generation: {e}")
            raise # Fail the activity for workflow handling
        except Exception as e:
            activity.logger.error(f"Unexpected error during embedding generation: {e}")
            raise

    @activity.defn
    async def llm_generate(self, model: str, prompt: str, system: Optional[str] = None, format_hint: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates text using the specified OpenAI chat model, optionally with a system prompt and JSON formatting.

        Args:
            model: The OpenAI chat model to use (e.g., "gpt-3.5-turbo").
            prompt: The user prompt for the model.
            system: An optional system message to guide the model's behavior.
            format_hint: Hints for the response format (e.g., {'type': 'json_object'}).

        Returns:
            A dictionary containing the parsed LLM response (expected to be a list of dicts)
            under the 'response' key. Returns an empty list if parsing fails.
        """
        activity.logger.info(f"Generating text with model '{model}', prompt length: {len(prompt)}")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request_params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        # Request JSON output if hinted
        if format_hint and format_hint.get("type") in ["json_object"]:
             activity.logger.info("Requesting JSON object format from OpenAI.")
             request_params["response_format"] = {"type": "json_object"}

        try:
            completion = self.openai_client.chat.completions.create(**request_params)
            response_content = completion.choices[0].message.content

            # Log the raw response for debugging
            activity.logger.info("--- RAW LLM RESPONSE START ---")
            activity.logger.info(response_content)
            activity.logger.info("--- RAW LLM RESPONSE END ---")

            if response_content is None:
                 activity.logger.warning("OpenAI returned None content. Returning empty list.")
                 parsed_response = []
            else:
                # Attempt to parse the response string (expected to be JSON)
                # The utility function handles logging errors internally
                parsed_response = parse_llm_json_response(response_content.strip(), activity.logger)

            # Return the parsed Python list (or empty list on failure)
            # Expected structure: {'response': [{}, {}, ...]}
            return {"response": parsed_response}

        except OpenAIError as e:
            activity.logger.error(f"OpenAI API error during chat completion: {e}")
            raise # Fail the activity
        except Exception as e:
            activity.logger.error(f"Unexpected error during chat completion: {e}")
            raise

    @activity.defn
    async def pinecone_query(self, query_embedding: List[float], n_results: int = 10, store_filter_tag: Optional[str] = None) -> Dict[str, List[List[Any]]]:
        """
        Queries the Pinecone index with an embedding vector, optionally filtering by store.

        Args:
            query_embedding: The embedding vector to query with.
            n_results: The maximum number of results to return.
            store_filter_tag: An optional store tag to filter results.

        Returns:
            A dictionary containing lists of document chunks and their corresponding metadata,
            structured as {'documents': [ [...] ], 'metadatas': [ [...] ]}.
            Returns empty lists if the query fails or finds no matches.
        """
        if not self.pinecone_index:
             activity.logger.error("Pinecone index is not initialized. Cannot query.")
             # Return empty results structure as expected by the workflow
             return {"documents": [[]], "metadatas": [[]]}

        if not query_embedding:
            activity.logger.warning(f"Received empty query embedding for index '{self.pinecone_index_name}'. Returning empty results.")
            return {"documents": [[]], "metadatas": [[]]}

        # Log vector details for debugging (optional, can be removed if too verbose)
        activity.logger.debug(f"Query vector type: {type(query_embedding)}, length: {len(query_embedding)}, sample: {query_embedding[:3]}...")

        filter_dict = {"store": store_filter_tag} if store_filter_tag else None
        log_filter_msg = f"with filter: {filter_dict}" if filter_dict else "with no filter"
        activity.logger.info(f"Querying index '{self.pinecone_index_name}', top_k={n_results} {log_filter_msg}")

        try:
            query_response: QueryResponse = self.pinecone_index.query(
                vector=query_embedding,
                top_k=n_results,
                filter=filter_dict,
                include_metadata=True,
                include_values=False # Embeddings typically not needed in response
            )

            documents = []
            metadatas = []
            if query_response.matches:
                activity.logger.info(f"Received {len(query_response.matches)} matches from Pinecone.")
                for match in query_response.matches:
                    metadata = match.metadata if isinstance(match.metadata, dict) else {}
                    documents.append(metadata.get('original_chunk', '')) # Default to empty string
                    metadatas.append(metadata) # Append full metadata
            else:
                 activity.logger.info("No matches found from Pinecone query.")

            # Return in the nested list format expected by the workflow
            return {"documents": [documents], "metadatas": [metadatas]}

        except Exception as e:
            # Log the error but return empty results to potentially allow workflow continuation
            activity.logger.error(f"Error querying Pinecone index '{self.pinecone_index_name}': {e}")
            return {"documents": [[]], "metadatas": [[]]}