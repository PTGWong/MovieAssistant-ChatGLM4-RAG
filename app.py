# 主应用模块，整合所有功能并提供Gradio界面

import gradio as gr
import pandas as pd
import os
from typing import Tuple, Dict, List, Any

from tmdb_data import get_movies_batch, create_movies_dataframe, save_movies_to_csv, load_movies_from_csv
from vector_store import store_movies_in_chroma, query_similar_movies
from llm_service import generate_movie_recommendations
from config import MAX_RECOMMENDATIONS


def fetch_and_store_movies(num_movies: int = 100) -> str:
    """
    获取电影数据并存储到向量数据库
    
    Args:
        num_movies: 要获取的电影数量
        
    Returns:
        操作结果消息
    """
    try:
        
        if os.path.exists("movies_data.csv"):
            df_movies = load_movies_from_csv()
            if not df_movies.empty:
                message = f"已从本地文件加载 {len(df_movies)} 部电影数据。"
            else:
                
                movies = get_movies_batch(num_movies=num_movies)
                df_movies = create_movies_dataframe(movies)
                save_movies_to_csv(df_movies)
                message = f"已从TMDB获取并保存 {len(df_movies)} 部电影数据。"
        else:
            # 从TMDB获取电影数据
            movies = get_movies_batch(num_movies=num_movies)
            df_movies = create_movies_dataframe(movies)
            save_movies_to_csv(df_movies)
            message = f"已从TMDB获取并保存 {len(df_movies)} 部电影数据。"
        
        # 存储到向量数据库
        store_movies_in_chroma(df_movies)
        message += "\n电影数据已成功存储到向量数据库。"
        
        return message
    except Exception as e:
        return f"获取或存储电影数据时出错: {e}"


def recommend_movies(user_query: str) -> Tuple[str, str]:
    """
    根据用户查询推荐电影
    
    Args:
        user_query: 用户查询文本
        
    Returns:
        推荐电影列表和LLM生成的回复
    """
    try:
        # 查询相似电影
        movie_results = query_similar_movies(user_query, n_results=MAX_RECOMMENDATIONS)
        
        if not movie_results or "metadatas" not in movie_results or not movie_results["metadatas"]:
            return "未找到相关电影，请尝试其他查询。", ""
        
        result = generate_movie_recommendations(user_query, movie_results, max_recommendations=MAX_RECOMMENDATIONS)
        
        movie_details = ""
        for i, movie in enumerate(result["recommendations"], 1):
            movie_details += f"### {i}. {movie['title']}"
            if movie['original_title'] and movie['original_title'] != movie['title']:
                movie_details += f" ({movie['original_title']})"
            movie_details += "\n"
            
            movie_details += f"**上映日期**: {movie['release_date']}\n"
            movie_details += f"**类型**: {movie['genres']}\n"
            movie_details += f"**导演**: {movie['directors']}\n"
            movie_details += f"**主演**: {movie['cast']}\n"
            movie_details += f"**简介**: {movie['overview']}\n\n"
        
        return movie_details, result["llm_response"]
    except Exception as e:
        return f"推荐电影时出错: {e}", ""


def create_interface():
    """
    创建Gradio界面
    """
    with gr.Blocks(title="电影推荐系统") as interface:
        gr.Markdown("# 🎬 电影推荐系统")
        gr.Markdown("这是一个基于RAG和LLM的电影推荐系统，可以根据您的兴趣推荐电影。")
        
        with gr.Tab("初始化数据"):
            gr.Markdown("### 数据初始化")
            gr.Markdown("首次使用前，请先获取电影数据并存储到向量数据库。")
            
            with gr.Row():
                num_movies = gr.Slider(
                    minimum=10, 
                    maximum=500, 
                    value=100, 
                    step=10, 
                    label="要获取的电影数量"
                )
                fetch_button = gr.Button("获取并存储电影数据")
            
            init_output = gr.Textbox(label="初始化结果", lines=5)
            fetch_button.click(fetch_and_store_movies, inputs=[num_movies], outputs=[init_output])
        
        with gr.Tab("电影推荐"):
            gr.Markdown("### 电影推荐")
            gr.Markdown("输入您的电影偏好，系统将为您推荐相关电影。")
            
            user_query = gr.Textbox(label="您想看什么类型的电影？", placeholder="例如：我想看一部关于太空探索的科幻电影")
            recommend_button = gr.Button("获取推荐")
            
            with gr.Row():
                with gr.Column():
                    movie_details = gr.Markdown(label="推荐电影列表")
                with gr.Column():
                    llm_response = gr.Markdown(label="AI推荐理由")
            
            recommend_button.click(
                recommend_movies, 
                inputs=[user_query], 
                outputs=[movie_details, llm_response]
            )
    
    return interface


if __name__ == "__main__":
    # 创建并启动Gradio
    interface = create_interface()
    interface.launch()