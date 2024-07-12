import argparse
import asyncio
from graph_search import setup_global_search, setup_local_search, create_search_engine
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

async def query_graphrag(search_engine, query):
    result = await search_engine.asearch(query)
    return result.response

# Example usage
clearance_hierarchy = {
    "general": ["general"],
    "level_q": ["general", "level_q"],
    "admin": ["general", "level_q", "admin"]
}

def get_most_recent_folder(folder_path):
    import os
    # extract only the name of the folder with the largest hyphenated number
    folder_list = [folder for folder in os.listdir(folder_path) if "-" in folder]
    # sort first by the number after the hyphen, then by the number after the hyphen
    folder_list.sort(key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1])))
    # get the largest number folder
    largest_folder = folder_list[-1]
    return os.path.join(folder_path, largest_folder)

def format_responses(responses, query):
    prompt = f"Please synthesize the following responses to answer the question:\n\n"
    for i, response in enumerate(responses, 1):
        prompt += f"Response {i}:\n{response}\n\n"
    prompt += f"Question: {query}"
    prompt += f"Answer: "
    return prompt


async def main():
    import os
    base_dir = "./graph_rag_indexes"
    dirs = ['general', 'level_q', 'admin']
    input_dirs = {}
    for direc in dirs:
        folder_path = os.path.join(os.path.join(base_dir, direc), 'output')
        input_dirs[direc] = os.path.join(get_most_recent_folder(folder_path),'artifacts')
    
    
    # user_clearance = "admin"

    parser = argparse.ArgumentParser(description='Run query as different user clearance levels.')
    parser.add_argument('clearance_type', type=str, help='Clearance type (e.g., general, level_q, admin)')
    args = parser.parse_args()
    user_clearance = args.clearance_type or ''



    search_type = "global"

    context_builders = []
    token_encoders = []

    for level in clearance_hierarchy[user_clearance]:
        input_dir = input_dirs[level]
        if search_type == "global":
            context_builder, token_encoder = setup_global_search(input_dir, community_level=2)
        else:
            context_builder, token_encoder = setup_local_search(input_dir, f"{input_dir}/lancedb", community_level=2)
        print(f"Loaded -- {level} -- context builder and token encoder")
        context_builders.append(context_builder)
        token_encoders.append(token_encoder)

    responses = []
    query = "What kind of diet do you need to train a dragon?"
    for i, context_builder in enumerate(context_builders):
        # Choose the appropriate context_builder and token_encoder based on the highest clearance level available
        search_engine = create_search_engine(context_builder, token_encoders[i], search_type=search_type)
        
        response = await query_graphrag(search_engine, query)
        level = clearance_hierarchy[user_clearance][i]
        print(f"Response from {level} search: {response}\n")
        responses.append(response)
    
    
    # synthesize all responses into a single response using ChatGPT from LangChain
    formatted_prompt = format_responses(responses, query)

    llm = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ["OPENAI_API_MODEL"],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that synthesizes information and give relevant information back to the user. You are also honest if you don't know the answer to a question. If the question doesn't need specific data and can be answered with general knowledge, you can answer it. If the question needs specific data, you can ask for clarification. If the question is too broad, you can ask for more details. If the question is inappropriate, you can refuse to answer.",
            ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm
    answer = chain.invoke({
        "input": formatted_prompt
    })

    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
