from openai import OpenAI
from llama_index import StorageContext, load_index_from_storage
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index import ServiceContext, set_global_service_context, VectorStoreIndex, SimpleDirectoryReader
from utils import *

#APIs for calling service

def query(text):
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(  api_key=openai_api_key, base_url=openai_api_base )

    prompts=text
    completion = client.completions.create(model="../vLLM/hf_hub/llama2-7b",
                                      prompt=prompts,
                                      max_tokens=1024)
    return completion.choices[0].text


def query_with_rag(text):
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model,llm=None)
    # optionally set a global service context
    set_global_service_context(service_context)

     # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(  api_key=openai_api_key, base_url=openai_api_base )

    storage_context = StorageContext.from_defaults(persist_dir="D:/DATA/brain/STORAGE")
    index = load_index_from_storage(storage_context)
    
    query_engine = index.as_query_engine()
    response=query_engine.query(text)
    
    prompts=response.response
    completion = client.completions.create(model="../vLLM/hf_hub/llama2-7b",
                                      prompt=prompts,
                                      max_tokens=1280,
                                      
                                      )
    
    paper_ids=extract_id_from_meta(response.metadata)
    
    titles=[get_title(id) for id in paper_ids]
    
    
    return titles,prompts,completion.choices[0].text
    

def query_figure(text):
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model,llm=None)
    # optionally set a global service context
    set_global_service_context(service_context)

    storage_context = StorageContext.from_defaults(persist_dir="D:/DATA/brain/CAPTION/storage")
    index = load_index_from_storage(storage_context)
    
    query_engine = index.as_query_engine()
    response=query_engine.query(text)
    figure_list=extract_figure_from_meta(response.metadata)
    caption=response.response
    
    
    return figure_list,caption

def query_context(message):
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model,llm=None)
    # optionally set a global service context
    set_global_service_context(service_context)

    storage_context = StorageContext.from_defaults(persist_dir="D:/DATA/brain/STORAGE")
    index = load_index_from_storage(storage_context)
    
    query_engine = index.as_query_engine()
    response=query_engine.query(message)
    
    paper_ids=extract_id_from_meta(response.metadata)
    
    titles=[get_title(id) for id in paper_ids]
    
    
    return titles,response.response





# if __name__ == '__main__':
#     prompt="What is diffusion tensor imaging?"
#     context,answer=query_with_rag(prompt)
    
#     print(context)
#     print(answer)
    
