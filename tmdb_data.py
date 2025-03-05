# 从TMDB API获取电影数据的模块

import requests
import pandas as pd
from typing import List, Dict, Any, Optional
import time
from config import TMDB_API_KEY

# TMDB API的URL
BASE_URL = "https://api.themoviedb.org/3"


def get_popular_movies(page: int = 1, language: str = "zh-CN") -> Dict[str, Any]:
    """
    获取热门电影列表
    
    Args:
        page: 页码
        language: 语言代码
        
    Returns:
        包含热门电影信息的字典
    """
    endpoint = f"{BASE_URL}/movie/popular"
    params = {
        "api_key": TMDB_API_KEY,
        "language": language,
        "page": page
    }
    
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"获取热门电影失败: {response.status_code}")
        return {"results": []}


def get_movie_details(movie_id: int, language: str = "zh-CN") -> Dict[str, Any]:
    """
    获取电影详细信息
    
    Args:
        movie_id: 电影ID
        language: 语言代码
        
    Returns:
        包含电影详细信息的字典
    """
    endpoint = f"{BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "language": language,
        "append_to_response": "credits,keywords"
    }
    
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"获取电影详情失败: {response.status_code}")
        return {}


def search_movies(query: str, language: str = "zh-CN", page: int = 1) -> Dict[str, Any]:
    """
    搜索电影
    
    Args:
        query: 搜索关键词
        language: 语言代码
        page: 页码
        
    Returns:
        包含搜索结果的字典
    """
    endpoint = f"{BASE_URL}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "language": language,
        "query": query,
        "page": page
    }
    
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"搜索电影失败: {response.status_code}")
        return {"results": []}


def get_movies_batch(num_movies: int = 100, language: str = "zh-CN") -> List[Dict[str, Any]]:
    """
    批量获取电影数据
    
    Args:
        num_movies: 要获取的电影数量
        language: 语言代码
        
    Returns:
        包含电影详细信息的列表
    """
    movies = []
    pages_needed = (num_movies + 19) // 20  # 每页20部电影
    
    for page in range(1, pages_needed + 1):
        popular_movies = get_popular_movies(page=page, language=language)
        
        if "results" not in popular_movies:
            continue
            
        for movie in popular_movies["results"]:
            # 获取电影详情
            movie_details = get_movie_details(movie["id"], language=language)
            
            if movie_details:

                processed_movie = process_movie_data(movie_details)
                movies.append(processed_movie)
                
                if len(movies) >= num_movies:
                    return movies
            
            # 添加延迟以避免API速率限制
            time.sleep(0.25)
    
    return movies


def process_movie_data(movie: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理电影数据，提取需要的字段
    
    Args:
        movie: 原始电影数据
        
    Returns:
        处理后的电影数据
    """
    # 导演信息
    directors = []
    if "credits" in movie and "crew" in movie["credits"]:
        directors = [crew["name"] for crew in movie["credits"]["crew"] if crew["job"] == "Director"]
    
    # 演员信息top 5
    cast = []
    if "credits" in movie and "cast" in movie["credits"]:
        cast = [actor["name"] for actor in movie["credits"]["cast"][:5]]
    
    keywords = []
    if "keywords" in movie and "keywords" in movie["keywords"]:
        keywords = [keyword["name"] for keyword in movie["keywords"]["keywords"]]
    
    # 电影类型
    genres = []
    if "genres" in movie:
        genres = [genre["name"] for genre in movie["genres"]]

    processed_movie = {
        "id": movie.get("id"),
        "title": movie.get("title"),
        "original_title": movie.get("original_title"),
        "overview": movie.get("overview"),
        "release_date": movie.get("release_date"),
        "popularity": movie.get("popularity"),
        "vote_average": movie.get("vote_average"),
        "vote_count": movie.get("vote_count"),
        "poster_path": movie.get("poster_path"),
        "backdrop_path": movie.get("backdrop_path"),
        "runtime": movie.get("runtime"),
        "genres": genres,
        "directors": directors,
        "cast": cast,
        "keywords": keywords,
        "language": movie.get("original_language"),
    }
    
    return processed_movie


def create_movies_dataframe(movies: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    将电影数据转换为DataFrame
    
    Args:
        movies: 电影数据列表
        
    Returns:
        包含电影数据的DataFrame
    """
    return pd.DataFrame(movies)


def save_movies_to_csv(df: pd.DataFrame, file_path: str = "movies_data.csv") -> None:
    """
    将电影数据保存为CSV文件
    
    Args:
        df: 电影数据DataFrame
        file_path: 保存路径
    """
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"电影数据已保存至 {file_path}")


def load_movies_from_csv(file_path: str = "movies_data.csv") -> pd.DataFrame:
    """
    从CSV文件加载电影数据
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        包含电影数据的DataFrame
    """
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except Exception as e:
        print(f"加载电影数据失败: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # 测试代码
    movies = get_movies_batch(num_movies=100)
    df_movies = create_movies_dataframe(movies)
    print(f"获取了 {len(df_movies)} 部电影的数据")
    print(df_movies.columns)
    save_movies_to_csv(df_movies)