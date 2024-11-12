from joblib import Memory
memory = Memory(location='.', verbose=0)

def get_embedding(client, texts, model="text-embedding-3-small"):
    @memory.cache
    def get_cached_embedding(texts, model="text-embedding-3-small"):
        return client.embeddings.create(input = texts, model=model).data
    
    return get_cached_embedding(texts, model)