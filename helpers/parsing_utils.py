import json
from typing import List, Dict, Any
import logging # Use standard logging, the activity logger will be passed in

def parse_llm_json_response(response_content: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Parses a string expected to contain a JSON list of grocery items.

    Handles various formats the LLM might return:
    - A JSON list of objects: [{"name": "apple", "price": 1.00}, ...]
    - A single JSON object (if it represents one item): {"name": "apple", "price": 1.00}
    - A JSON object containing the list under a key: {"items": [{"name": "apple", ...}]}

    Args:
        response_content: The raw string content from the LLM response.
        logger: A logger instance for logging messages.

    Returns:
        A list of dictionaries, where each dictionary has at least 'name' and 'price' keys.
        Returns an empty list if parsing fails or the structure is incorrect.
    """
    parsed_response: List[Dict[str, Any]] = [] # Default to empty list
    if not response_content:
        logger.warning("Received empty response content for parsing.")
        return []

    # --- NEW: Strip potential markdown code fences --- START
    cleaned_content = response_content.strip()
    if cleaned_content.startswith("```json") and cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[len("```json"): -len("```")].strip()
        logger.info("Removed ```json markdown fences.")
    elif cleaned_content.startswith("```") and cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[len("```"): -len("```")].strip()
        logger.info("Removed ``` markdown fences.")
    # --- NEW: Strip potential markdown code fences --- END
    
    # Use the cleaned content for parsing
    if not cleaned_content:
        logger.warning("Content was empty after stripping fences.")
        return []

    try:
        # Use the cleaned content now
        parsed_obj = json.loads(cleaned_content)

        if isinstance(parsed_obj, list):
            # Handles [...] and []
            # Check if all items are dicts and have the required keys
            if all(isinstance(item, dict) and 'name' in item and 'price' in item for item in parsed_obj):
                parsed_response = parsed_obj
                logger.info(f"Successfully parsed LLM response into a list of {len(parsed_response)} dicts.")
            else:
                logger.warning("Parsed JSON list did not contain only dictionaries with 'name' and 'price' keys. Returning empty list.")
                parsed_response = [] # Keep strict: require name/price

        elif isinstance(parsed_obj, dict):
            # Handles the case where LLM returns just one item as an object {}
            if 'name' in parsed_obj and 'price' in parsed_obj:
                logger.info("Parsed LLM response as a single dictionary, wrapping it in a list.")
                parsed_response = [parsed_obj]
            else:
                # Check if dict contains a key whose value is the desired list of dicts
                found_list_in_dict = False
                for key, value in parsed_obj.items():
                    # Check if the value is a list, and all its items are dicts with 'name' and 'price'
                    if isinstance(value, list) and all(isinstance(item, dict) and 'name' in item and 'price' in item for item in value):
                        logger.info(f"Found expected list of dicts under key '{key}' in the response object.")
                        parsed_response = value
                        found_list_in_dict = True
                        break # Use the first suitable list found
                
                if not found_list_in_dict:
                   logger.warning("Parsed dictionary object was not a single item with name/price and did not contain a recognized nested list of items. Returning empty list.")
                   parsed_response = []

        else:
            # If it's not a list or a dict we can handle, log warning and return empty
            logger.warning(f"LLM JSON response was not a list or a recognized dictionary structure (Type: {type(parsed_obj)}). Returning empty list.")
            parsed_response = [] # Explicitly set default

    except json.JSONDecodeError as json_err:
        # Log the *original* content for easier debugging if parsing fails
        logger.error(f"Failed to parse LLM response string as JSON: {json_err}. Original Response: {response_content}")
        parsed_response = [] # Ensure [] on JSON error
    except Exception as e:
         logger.error(f"Unexpected error during final response parsing: {e}. Returning empty list.")
         parsed_response = [] # Ensure [] on other errors

    return parsed_response 