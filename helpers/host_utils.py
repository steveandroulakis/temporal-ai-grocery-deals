import os
import yaml
import logging
from typing import Optional

# Add necessary imports for the new function
from pinecone.grpc import PineconeGRPC as Pinecone, GRPCClientConfig
from pinecone import Index as PineconeIndex # Use alias consistent with activities
# Path to docker-compose relative to workspace root
DOCKER_COMPOSE_PATH = "pinecone/docker-compose.yaml"

def determine_pinecone_host(index_name: str, docker_compose_path: str, logger: logging.Logger) -> str:
    """Helper method to determine Pinecone host from docker-compose."""
    # Find docker-compose path relative to likely workspace root
    docker_compose_abs_path = os.path.abspath(docker_compose_path)
    if not os.path.exists(docker_compose_abs_path):
        # Try path relative to this script's location if absolute fails
        script_dir = os.path.dirname(__file__)
        # Go up one level (from helpers to root) then to the relative docker_compose_path
        rel_path = os.path.join(script_dir, '..', docker_compose_path)
        docker_compose_abs_path = os.path.abspath(rel_path)
        if not os.path.exists(docker_compose_abs_path):
            logger.error(f"docker-compose.yaml not found at {docker_compose_path} or relative path. Cannot determine port.")
            raise FileNotFoundError(f"docker-compose.yaml not found (checked {docker_compose_path} and {rel_path}), cannot determine Pinecone port.")

    # Load docker-compose and extract port
    pinecone_grpc_host: Optional[str] = None
    try:
        with open(docker_compose_abs_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config or 'services' not in config or index_name not in config['services']:
            logger.error(f"Invalid docker-compose file ({docker_compose_abs_path}) or service '{index_name}' not found.")
            raise ValueError(f"Invalid docker-compose file or service '{index_name}' not found.")
        service_config = config['services'][index_name]
        if 'ports' in service_config and service_config['ports']:
            # Assuming format 'HOST_PORT:CONTAINER_PORT'
            port_mapping = str(service_config['ports'][0]) # Ensure it's a string
            host_port_str = port_mapping.split(':')[0]
            if host_port_str.isdigit():
                port = int(host_port_str) # Get the host port
                pinecone_grpc_host = f"localhost:{port}"
                logger.info(f"Determined Pinecone gRPC host from docker-compose ({docker_compose_abs_path}): {pinecone_grpc_host}")
            else:
                 raise ValueError(f"Could not parse host port from mapping '{port_mapping}' for service '{index_name}'. Expected format 'HOST:CONTAINER'.")
        else:
            raise ValueError(f"No 'ports' mapping found for service '{index_name}' in {docker_compose_abs_path}.")
    except (yaml.YAMLError, ValueError, IndexError, TypeError, FileNotFoundError) as e:
        logger.error(f"Error reading docker-compose ({docker_compose_abs_path}) or extracting port for '{index_name}': {e}")
        raise

    if not pinecone_grpc_host:
         # This case should ideally be unreachable if exceptions are raised correctly above
         logger.error(f"Failed to determine pinecone_grpc_host for '{index_name}' from {docker_compose_abs_path}.")
         raise ValueError(f"Failed to determine pinecone_grpc_host for '{index_name}'.")

    return pinecone_grpc_host 

# NEW function moved from DealFinderActivities
def init_pinecone_index(pinecone_index_name: str, logger: logging.Logger) -> PineconeIndex:
    """Initializes and returns the Pinecone index client."""
    try:
        pinecone_grpc_host = determine_pinecone_host(
            index_name=pinecone_index_name,
            docker_compose_path=DOCKER_COMPOSE_PATH, # Use constant
            logger=logger
        )
        logger.info(f"Connecting to Pinecone index '{pinecone_index_name}' via gRPC at {pinecone_grpc_host}...")

        # Use a dummy key for local/plaintext gRPC connection
        pc = Pinecone(api_key="dummy-key", host=pinecone_grpc_host, plaintext=True)
        index = pc.Index(
            name=pinecone_index_name,
            host=pinecone_grpc_host,
            grpc_config=GRPCClientConfig(secure=False) # Assuming local/insecure connection
        )

        # Simple check to confirm connection (optional: add describe_index_stats if needed)
        if not hasattr(index, 'query'):
             raise TypeError("Failed to obtain a valid Pinecone index object for querying.")

        logger.info(f"Successfully obtained gRPC index handle for '{pinecone_index_name}'.")
        return index

    except Exception as e:
        logger.error(f"Error initializing Pinecone index '{pinecone_index_name}': {e}")
        raise 