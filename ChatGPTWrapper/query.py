import asyncio
from graph_search import setup_global_search, setup_local_search, create_search_engine

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


async def main():
    import os
    base_dir = "./graph_rag_indexes"
    dirs = ['general', 'level_q', 'admin']
    input_dirs = {}
    for direc in dirs:
        folder_path = os.path.join(os.path.join(base_dir, direc), 'output')
        input_dirs[direc] = os.path.join(get_most_recent_folder(folder_path),'artifacts')
    
    
    user_clearance = "level_q"
    search_type = "global"

    context_builders = []
    token_encoders = []

    for level in clearance_hierarchy[user_clearance]:
        input_dir = input_dirs[level]
        if search_type == "global":
            context_builder, token_encoder = setup_global_search(input_dir, community_level=2)
        else:
            context_builder, token_encoder = setup_local_search(input_dir, f"{input_dir}/lancedb", community_level=2)
        
        context_builders.append(context_builder)
        token_encoders.append(token_encoder)

    # Choose the appropriate context_builder and token_encoder based on the highest clearance level available
    search_engine = create_search_engine(context_builders[-1], token_encoders[-1], search_type=search_type)
    
    query = "What kind of diet do you need to train a dragon?"
    response = await query_graphrag(search_engine, query)
    
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
