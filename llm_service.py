# LLM服务模块，负责调用智谱ChatGLM4 API进行大模型推理

import requests
import json
from typing import List, Dict, Any, Optional
from config import ZHIPU_API_KEY, LLM_MODEL


def call_llm_api(messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    """
    调用智谱ChatGLM4 API进行对话
    
    Args:
        messages: 对话历史消息列表
        temperature: 温度参数，控制随机性
        
    Returns:
        LLM的回复文本
    """
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZHIPU_API_KEY}"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                print("API返回结果格式异常")
                return "抱歉，我无法处理您的请求。"
        else:
            print(f"调用LLM API失败: {response.status_code}, {response.text}")
            return "抱歉，我无法处理您的请求。"
    except Exception as e:
        print(f"调用LLM API时出错: {e}")
        return "抱歉，我无法处理您的请求。"


def generate_movie_recommendations(query: str, movie_results: Dict[str, Any], max_recommendations: int = 5) -> Dict[str, Any]:
    """
    根据用户查询和检索到的电影结果生成推荐回复
    
    Args:
        query: 用户查询
        movie_results: 从向量数据库检索到的电影结果
        max_recommendations: 最大推荐数量
        
    Returns:
        包含推荐电影和LLM回复的字典
    """
    
    recommendations = []
    seen_titles = set()
    
    if movie_results and "metadatas" in movie_results and movie_results["metadatas"]:
        for metadata in movie_results["metadatas"][0]:
            title = metadata.get("title", "未知标题")
            if title not in seen_titles and title != "":
                seen_titles.add(title)
                
                recommendations.append({
                    "title": title,
                    "original_title": metadata.get("original_title", ""),
                    "overview": metadata.get("overview", "暂无简介"),
                    "release_date": metadata.get("release_date", "未知上映日期"),
                    "genres": metadata.get("genres", ""),
                    "directors": metadata.get("directors", ""),
                    "cast": metadata.get("cast", "")
                })
                
                if len(recommendations) >= max_recommendations:
                    break
    
    # 提示词
    system_prompt = """你是一个专业的电影推荐助手，根据用户的查询和检索到的电影信息，提供个性化的电影推荐。
你的回复应该包括：
1. 对用户查询的理解和分析
2. 基于用户兴趣推荐的电影列表
3. 每部电影的简要介绍和推荐理由
请确保你的回复友好、专业且有帮助。不要推荐重复的电影。"""
    
    # 构建电影信息文本
    movies_text = ""
    for i, movie in enumerate(recommendations, 1):
        movies_text += f"电影{i}: {movie['title']}\n"
        if movie['original_title'] and movie['original_title'] != movie['title']:
            movies_text += f"原标题: {movie['original_title']}\n"
        movies_text += f"上映日期: {movie['release_date']}\n"
        movies_text += f"类型: {movie['genres']}\n"
        movies_text += f"导演: {movie['directors']}\n"
        movies_text += f"主演: {movie['cast']}\n"
        movies_text += f"简介: {movie['overview']}\n\n"
    
    user_message = f"用户查询: {query}\n\n检索到的电影信息:\n{movies_text}\n请根据用户查询和检索到的电影信息，生成个性化的电影推荐回复。"
    
    # 调用LLM API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    llm_response = call_llm_api(messages, temperature=0.7)
    
    return {
        "recommendations": recommendations,
        "llm_response": llm_response
    }


if __name__ == "__main__":
    # 测试代码
    from vector_store import query_similar_movies
    
    test_query = "我想看一部关于太空探索的科幻电影"
    movie_results = query_similar_movies(test_query)
    
    if movie_results:
        result = generate_movie_recommendations(test_query, movie_results)
        print(result["llm_response"])
    else:
        print("未找到相关电影")