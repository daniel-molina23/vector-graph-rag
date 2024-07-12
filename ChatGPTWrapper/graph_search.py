import os
import pandas as pd
import tiktoken
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import read_indexer_covariates, read_indexer_relationships, read_indexer_text_units
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from dotenv import load_dotenv

load_dotenv()

# Load community reports and entities for Global Search
def setup_global_search(input_dir, community_level):
    community_report_table = "create_final_community_reports"
    entity_table = "create_final_nodes"
    entity_embedding_table = "create_final_entities"

    entity_df = pd.read_parquet(f"{input_dir}/{entity_table}.parquet")
    report_df = pd.read_parquet(f"{input_dir}/{community_report_table}.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/{entity_embedding_table}.parquet")

    reports = read_indexer_reports(report_df, entity_df, community_level)
    entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)
    
    token_encoder = tiktoken.get_encoding("cl100k_base")

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )

    return context_builder, token_encoder

# Load text units, entities, relationships, and covariates for Local Search
def setup_local_search(input_dir, lancedb_uri, community_level):
    community_report_table = "create_final_community_reports"
    entity_table = "create_final_nodes"
    entity_embedding_table = "create_final_entities"
    relationship_table = "create_final_relationships"
    covariate_table = "create_final_covariates"
    text_unit_table = "create_final_text_units"

    entity_df = pd.read_parquet(f"{input_dir}/{entity_table}.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/{entity_embedding_table}.parquet")
    relationship_df = pd.read_parquet(f"{input_dir}/{relationship_table}.parquet")
    covariate_df = pd.read_parquet(f"{input_dir}/{covariate_table}.parquet")
    text_unit_df = pd.read_parquet(f"{input_dir}/{text_unit_table}.parquet")

    entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)
    relationships = read_indexer_relationships(relationship_df)
    claims = read_indexer_covariates(covariate_df)
    covariates = {"claims": claims}
    text_units = read_indexer_text_units(text_unit_df)

    description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
    description_embedding_store.connect(db_uri=lancedb_uri)
    store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

    token_encoder = tiktoken.get_encoding("cl100k_base")
    text_embedder = OpenAIEmbedding(
        api_key=os.environ["GRAPHRAG_API_KEY"],
        model=os.environ["GRAPHRAG_EMBEDDING_MODEL"],
        api_type=OpenaiApiType.OpenAI,
    )

    context_builder = LocalSearchMixedContext(
        community_reports=read_indexer_reports(pd.read_parquet(f"{input_dir}/{community_report_table}.parquet"), entity_df, community_level),
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    return context_builder, token_encoder


def create_search_engine(context_builder, token_encoder, search_type="global"):
    llm = ChatOpenAI(
        api_key=os.environ["GRAPHRAG_API_KEY"],
        model=os.environ["GRAPHRAG_LLM_MODEL"],
        api_type=OpenaiApiType.OpenAI,
    )

    if search_type == "global":
        context_builder_params = {
            "use_community_summary": False,
            "shuffle_data": True,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            "max_tokens": 12000,
            "context_name": "Reports",
        }

        map_llm_params = {
            "max_tokens": 1000,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        reduce_llm_params = {
            "max_tokens": 2000,
            "temperature": 0.0,
        }

        search_engine = GlobalSearch(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            max_data_tokens=12000,
            map_llm_params=map_llm_params,
            reduce_llm_params=reduce_llm_params,
            allow_general_knowledge=False,
            json_mode=True,
            context_builder_params=context_builder_params,
            concurrent_coroutines=32,
            response_type="multiple paragraphs",
        )

    else:
        local_context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,
            "max_tokens": 12000,
        }

        llm_params = {
            "max_tokens": 2000,
            "temperature": 0.0,
        }

        search_engine = LocalSearch(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=local_context_params,
            response_type="multiple paragraphs",
        )

    return search_engine
