# Graph-Powered Movie Intelligence Agent

## 🛠️ Tech Stack
Database: Neo4j (Graph Database)

Orchestration: LangChain (Advanced LCEL & Custom Agents)

Interface: Streamlit

LLM: Google Gemini 

## 📖 The Challenge (The "Why")
Standard LLMs often suffer from "hallucinations" and lack specific, up-to-date knowledge about deep movie catalogs or user-specific preferences. I wanted to build a recommendation engine that didn't just guess, but actually "reasoned" through actor-director relationships and historical user data to provide high-fidelity suggestions.

## 💡 The Solution: GraphRAG
Instead of a simple vector search, I implemented a GraphRAG (Graph Retrieval-Augmented Generation) approach. By leveraging Neo4j, the assistant can traverse complex nodes (Actors, Genres, Directors) to find non-obvious connections that a standard database would miss.

Key Feature: Personalized recommendations based on historical user ratings stored within the graph.

Interface: A conversational chat UI built with Streamlit for real-time interaction.

## 🧠 The "Hard Part": Beyond AgentExecutor
The most significant technical hurdle was migrating away from legacy LangChain classes like create_react_agent and AgentExecutor.

The Problem: Legacy executors were "black boxes" that made it difficult to handle complex state or custom error recovery during long graph queries.

The Fix: I refactored the entire logic into a more granular, controllable architecture (using LCEL/LangGraph). This allowed for better debugging, more predictable agent behavior, and a significantly more responsive user experience.

## 🚀 Key Impact Points (The "Result")
Increased Accuracy: Eliminated factual errors regarding movie credits by forcing the LLM to validate against the Neo4j schema.

Future-Proofed: Modernized the codebase to support complex, multi-step reasoning chains that legacy agents couldn't handle.

Context-Aware: Enabled the system to "remember" user tastes by querying the user-movie relationship nodes in the graph.

## Installation

### Install Python libraries

Firstly, install the required Python libraries. The quickest way to get up and running is to re-create the conda environment as follows:

```
conda env create -f environment.yml
source activate chatbot
```

### Setting Streamlit Secrets
To keep it secure, you will store the API key in the Streamlit secrets.toml file.

Create a new file, .streamlit/secrets.toml, and copy the following text, adding your OpenAI and Google API keys:

```
OPENAI_API_KEY = "sk-...."
OPENAI_MODEL = "gpt-4"

GOOGLE_API_KEY = "...."
GEMINI_MODEL = 'gemini-3-flash-preview'
```

### Create a Neo4j cloud instance. 

Neo4j Aura is Neo4j’s fully managed cloud service.

You can run the GDS plugin on a free Aura Instance right out of the box—no payment or card details needed.

Choose Aura if you:

Want to access your database from anywhere

Prefer not to install software locally

Have limited local storage or computing resources

Want automatic backups and updates

Setting up Neo4j Aura:

Navigate to AuraDB

Click **Start free** to create an account

![alt text](screenshots/Neo4j-startpage.png)

Neo4j Aura signup page with Start Free button highlighted

![alt text](screenshots/aura_signup_free_gds.png)

Follow the signup flow and select a free instance

Neo4j signup
You can destroy and spin up as many free instances as you like. Once you’re finished with this course, you can easily import your own data instead of the demo dataset.

To load the movie recommendations dataset:

Download this .dump file

Click Backup & Restore in your Aura instance

![alt text](screenshots/backup_from_dump.png)


Aura Instances page
Upload the .dump file

![alt text](screenshots/backup_from_dump_2.png)

Backup and restore tab in Aura Instances page
After the import completes, you’ll have the same dataset available in your own cloud instance.

Add the following secrets to your secrets.toml file:

```
NEO4J_URI = "..."
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "...."
NEO4J_DATABASE = "neo4j"
```

### Create the embeddings for the movie plots

Run the following script

`python create_plot_embeddings.py`

This will create a csv file of movie ids and plot embeddings.
I've uploaded this file to an AWS S3 bucket and created a sharable link. This is a workaround to quickly import and link these embeddings in Neo4j.

Run the following script in the Query 

```
LOAD CSV WITH HEADERS
FROM 'https://[S3-bucket].s3.eu-central-1.amazonaws.com/movie-plots-embeddings.csv[file_url]'
AS row
MATCH (m:Movie {movieId: row.movieId})
CALL db.create.setNodeVectorProperty(m, 'plotEmbedding', apoc.convert.fromJsonList(row.plotEmbedding));

CREATE VECTOR INDEX moviePlots IF NOT EXISTS
FOR (m:Movie)
ON m.plotEmbedding
OPTIONS {indexConfig: {
`vector.dimensions`: 1536,
`vector.similarity_function`: 'cosine'
}};

```

All set!

Now you can start the bot by runnning:
`streamlit run bot.py`







