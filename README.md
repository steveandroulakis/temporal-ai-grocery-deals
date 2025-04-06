# Temporal Deal Finder

Porting and enhancing Anthony's https://github.com/anthonywong555/temporal-grocery-search-deals

Work in progress.

* Search groceries workflow ([`deal_finder_workflow.py`](workflows/deal_finder_workflow.py)) uses ([`deal_finder_activities.py`](activities/deal_finder_activities.py)) activities for now.
* Sample grocery data ([`pinecone/grocery_data/`](pinecone/grocery_data/)) is loaded into Pinecone ([`pinecone/preload_vector_data.py`](pinecone/preload_vector_data.py)).

For more details see below.

## Configuration

This application uses `.env` files for configuration. Copy the [.env.example](.env.example) file to `.env` and update the values:

```bash
cp .env.example .env
```
* Only use this to load an OPENAI_API_KEY for now (for loading embeddings into Pinecone)

### LLM Provider Configuration

Must be openai for now (for loading embeddings into Pinecone)

Set the `LLM_PROVIDER` environment variable in your `.env` file to choose the desired provider:

- `LLM_PROVIDER=openai` for OpenAI's GPT-4o

## Configuring Temporal Connection

By default, this application will connect to a local Temporal server (`localhost:7233`) in the default namespace, using the `grocery-task-queue` task queue. You can override these settings in your `.env` file.

### Use Temporal Cloud

See [.env.example](.env.example) for details on connecting to Temporal Cloud using mTLS or API key authentication.

[Sign up for Temporal Cloud](https://temporal.io/get-cloud)

### Use a local Temporal Dev Server

On a Mac
```bash
brew install temporal
temporal server start-dev
```
See the [Temporal documentation](https://learn.temporal.io/getting_started/python/dev_environment/) for other platforms.

## Running the Application

This runs a Python version of the [Typescript-based retrieveFood workflow](https://github.com/anthonywong555/temporal-grocery-search-deals/blob/main/apps/worker/src/workflows/ai/retrieveFood.ts).

### Prerequisites

Requires [uv](https://github.com/astral-sh/uv) to manage dependencies and run commands within the project environment.

1. Ensure `uv` is installed.
2. Install project dependencies:
   ```bash
   # Installs dependencies from uv.lock into the virtual environment (.venv)
   uv sync
   ```
   If you need development dependencies (like pytest), install them with:
   ```bash
   uv sync --all-extras
   ```

#### Pinecone Vector Database

This application uses Pinecone for vector storage and retrieval (RAG). You need to run a local instance using Docker:

```bash
cd pinecone
docker compose -f pinecone/docker-compose.yaml up -d
```

Load the sample grocery data (located in `pinecone/grocery_data`):

```bash
# Runs the script within the uv-managed environment
uv run python pinecone/preload_vector_data.py
```

You can test the Retrieval-Augmented Generation (RAG) setup by searching for groceries:

```bash
uv run python pinecone/search_groceries.py "milk"
uv run python pinecone/search_groceries.py "citrus"
```

And you should see results!

### Running the Workflow

Run the following commands in separate terminal windows:

1. Start the Temporal worker:
```bash
uv run python -m scripts.run_worker
```

2. Start the Temporal workflow:
```bash
# Example query
uv run python -m scripts.run_workflow --query "pasta"
```

Example response:
```
Starting DealFinderWorkflow with ID: deal-finder-606afe74-c7e8-45c3-9675-8e584d898f80
Input Query: pasta
Indexes (Store Tags): ['Safeway', "Trader Joe's", 'Whole Foods']
Workflow started. Waiting for result...

--- Workflow Result ---
[
  {
    "collection": "Safeway",
    "results": [
      {
        "name": "Pasta, Spaghetti",
        "price": "1.29"
      }
    ]
  },
  {
    "collection": "Trader Joe's",
    "results": [
      {
        "name": "Pasta, Spaghetti",
        "price": "0.99"
      },
      {
        "name": "Pasta, Penne Rigate",
        "price": "0.99"
      },
      {
        "name": "Organic Brown Rice Pasta, Fusilli",
        "price": "2.99"
      }
    ]
  },
  {
    "collection": "Whole Foods",
    "results": [
      {
        "name": "Organic Whole Wheat Pasta",
        "price": "2.49"
      }
    ]
  }
]
```

You can also be fuzzy with your query:
```
# A more interesting query
uv run python -m scripts.run_workflow --query "to eat hummus with"
```

## TODO
Things I'm doing next
- [x] Combine grocery store vector data into a single vector index
- [x] Reduce cardinality of vector data
- [x] Replace Ollama calls with OpenAI calls in workflow
- [x] Ensure old Chroma DB activities are now working Pinecone ones
These will enable the Python port of the workflow to run.

Then I'll add the following:
- [ ] Schedule for updating vector data
- [ ] Notification workflow when a deal is found
- [ ] Figure out what logic determines the best deal
- [ ] Web UI (port of Anthony's existing one + chat functionality?)
- [ ] Other (currently under discussion!)