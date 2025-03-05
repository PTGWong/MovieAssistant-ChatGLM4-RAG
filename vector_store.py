# 向量存储模块，负责将电影数据转换为向量并存储到ChromaDB

import requests
import json
import pandas as pd
import chromadb
from typing import List, Dict, Any, Optional
from config import (
    SILICONFLOW_API_KEY, 
    SILICONFLOW_API_URL, 
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY
)


def get_embedding(text: str) -> List[float]:
    """
    使用硅基API获取文本的embedding向量
    
    Args:
        text: 需要转换为向量的文本
        
    Returns:
        文本的embedding向量
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}"
    }
    
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    
    try:
        response = requests.post(
            f"{SILICONFLOW_API_URL.replace('chat/completions', 'embeddings')}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["data"][0]["embedding"]
        else:
            print(f"获取embedding失败: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        print(f"获取embedding时出错: {e}")
        return []


def get_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """
    批量获取文本的embedding向量
    
    Args:
        texts: 需要转换为向量的文本列表
        
    Returns:
        文本的embedding向量列表
    """
    embeddings = []
    
    for text in texts:
        if not text or text.strip() == "":
            
            embeddings.append([0.0] * 1024)  # BGE-M3模型的向量维度为1024
            continue
            
        embedding = get_embedding(text)
        if embedding:
            embeddings.append(embedding)
        else:
            embeddings.append([0.0] * 1024)
    
    return embeddings


def process_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    处理元数据，确保所有值都是字符串类型
    
    Args:
        metadata: 原始元数据
        
    Returns:
        处理后的元数据
    """
    processed = {}
    for key, value in metadata.items():
        if value is None:
            processed[key] = ""
        elif isinstance(value, list):
            processed[key] = ", ".join(map(str, value))
        elif isinstance(value, dict):
            processed[key] = json.dumps(value, ensure_ascii=False)
        else:
            processed[key] = str(value)
    return processed


def create_chroma_client():
    """
    创建ChromaDB客户端
    
    Returns:
        ChromaDB客户端实例
    """
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)


def get_or_create_collection(client):
    """
    获取或创建ChromaDB集合
    
    Args:
        client: ChromaDB客户端
        
    Returns:
        ChromaDB集合
    """
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


def store_movies_in_chroma(df_movies: pd.DataFrame) -> None:
    """
    将电影数据存储到ChromaDB
    
    Args:
        df_movies: 包含电影数据的DataFrame
    """
    client = create_chroma_client()
    collection = get_or_create_collection(client)
    
    # 准备数据
    df_unique = df_movies.drop_duplicates(subset=['id'])
    ids = [str(movie_id) for movie_id in df_unique["id"].tolist()]
    documents = df_unique["overview"].fillna("").tolist()
    
    # 获取文档的embedding
    embeddings = get_batch_embeddings(documents)
    
    metadatas = df_unique.to_dict("records")
    processed_metadatas = [process_metadata(metadata) for metadata in metadatas]
    
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        batch_ids = ids[i:end]
        batch_documents = documents[i:end]
        batch_embeddings = embeddings[i:end]
        batch_metadatas = processed_metadatas[i:end]
        
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
    
    print(f"成功将 {len(ids)} 部电影数据存储到ChromaDB")


def query_similar_movies(query_text: str, n_results: int = 5):
    """
    查询与输入文本相似的电影
    
    Args:
        query_text: 查询文本
        n_results: 返回结果数量
        
    Returns:
        相似电影的结果
    """
    client = create_chroma_client()
    collection = get_or_create_collection(client)
    
    # embedding
    query_embedding = get_embedding(query_text)
    
    if not query_embedding:
        print("获取查询embedding失败")
        return []
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    return results


if __name__ == "__main__":
    # 测试代码
    from tmdb_data import load_movies_from_csv
    
    df_movies = load_movies_from_csv()
    
    if not df_movies.empty:
        # 存储到ChromaDB
        store_movies_in_chroma(df_movies)
        
        # 测试查询
        test_query = "科幻电影，有关于太空探索"
        results = query_similar_movies(test_query)
        
        if results and "metadatas" in results and results["metadatas"]:
            print(f"查询: {test_query}")
            for i, metadata in enumerate(results["metadatas"][0]):
                print(f"结果 {i+1}: {metadata.get('title')} - {metadata.get('overview')[:100]}...")