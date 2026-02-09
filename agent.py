from llm import llm, embeddings
from graph import graph
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import tool
from langchain_neo4j import Neo4jVector, GraphCypherQAChain
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from utils import get_session_id


neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              
    graph=graph,                            
    index_name="moviePlots",                 
    node_label="Movie",                      
    text_node_property="plot",               
    embedding_node_property="plotEmbedding", 
    retrieval_query="""
    RETURN
        node.plot AS text,
        score,
        {
            title: node.title,
            directors: [ (person)-[:DIRECTED]->(node) | person.name ],
            actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
            tmdbId: node.tmdbId,
            source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
        } AS metadata
    """
    )


@tool("MoviePlotSearch")
def get_movie_plot(input: str) -> str:

    """For when you need to find information about movies based on a plot"""

    

    # Create the retriever

    retriever = neo4jvector.as_retriever()


    # Create the prompt

    instructions = (
        "Use the given context to answer the question."
        "If you don't know the answer, say you don't know."
        "Context: {context}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", instructions),
        ("human", "{input}")
        ]
    )

    # Create the chain 

    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(input)

@tool("MovieInformation")
def cypher_qa(input: str) -> str:

    """Provide information about movie questions using Cypher"""

    CYPHER_GENERATION_TEMPLATE = """
    You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
    Convert the user's question based on the schema.

    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.

    Do not return entire nodes or embedding properties.

    Fine Tuning:

    For movie titles that begin with "The", move "the" to the end. For example "The 39 Steps" becomes "39 Steps, The" or "the matrix" becomes "Matrix, The".

    Example Cypher Statements:

    1. To find who acted in a movie:
    ```
    MATCH (p:Person)-[r:ACTED_IN]->(m:Movie {{title: "Movie Title"}})
    RETURN p.name, r.role
    ```

    2. To find who directed a movie:
    ```
    MATCH (p:Person)-[r:DIRECTED]->(m:Movie {{title: "Movie Title"}})
    RETURN p.name
    ```

    3. How to find how many degrees of separation there are between two people:
    ```
    MATCH path = shortestPath(
      (p1:Person {{name: "Actor 1"}})-[:ACTED_IN|DIRECTED*]-(p2:Person {{name: "Actor 2"}})
    )
    WITH path, p1, p2, relationships(path) AS rels
    RETURN
      p1 {{ .name, .born, link:'https://www.themoviedb.org/person/'+ p1.tmdbId }} AS start,
      p2 {{ .name, .born, link:'https://www.themoviedb.org/person/'+ p2.tmdbId }} AS end,
      reduce(output = '', i in range(0, length(path)-1) |
        output + CASE
          WHEN i = 0 THEN
           startNode(rels[i]).name + CASE WHEN type(rels[i]) = 'ACTED_IN' THEN ' played '+ rels[i].role +' in 'ELSE ' directed ' END + endNode(rels[i]).title
           ELSE
             ' with '+ startNode(rels[i]).name + ', who '+ CASE WHEN type(rels[i]) = 'ACTED_IN' THEN 'played '+ rels[i].role +' in '
        ELSE 'directed '
          END + endNode(rels[i]).title
          END
      ) AS pathBetweenPeople
    ```

    Schema:
    {schema}

    Question:
    {question}
    """

    cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

    # Create the Cypher QA chain

    chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=True,
        cypher_prompt=cypher_prompt,
        allow_dangerous_requests=True
    )

    return chain.invoke(input)


@tool("GeneralChat")
def movie_chat(input: str) -> str:
    """For general movie chat not covered by other tools"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a movie expert providing information about movies."),
        ("human", "{input}"),
    ])

    chain = chat_prompt | llm | StrOutputParser()
    
    # You must call .invoke() with the required input dictionary
    return chain.invoke({"input": input})


agent_instructions = """
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors or directors.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, just output the final answer

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

chat_agent = create_agent(
    llm,
    tools = [get_movie_plot,cypher_qa,movie_chat],
    system_prompt=agent_instructions,
    checkpointer=InMemorySaver(),
    debug=True
)



def generate_response(user_input):

    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        {"configurable": {"thread_id": get_session_id()}},)

    return response['messages'][-1].content[0]['text']

