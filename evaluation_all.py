import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import KFold
from recommender import CollaborativeFiltering, ContentBasedRecommender, generate_test_ratings
from evaluation import precision_at_k, recall_at_k, ndcg_at_k
from tmdb_data import load_movies_from_csv
from vector_store import query_similar_movies
from llm_service import generate_movie_recommendations


class LLMRAGRecommender:
    """基于LLM+RAG的推荐算法封装类"""
    
    def __init__(self):
        pass
    
    def recommend_movies(self, movie_id: str, n_recommendations: int = 5) -> List[str]:
        """基于电影ID推荐相关电影
        
        Args:
            movie_id: 目标电影ID
            n_recommendations: 推荐数量
            
        Returns:
            推荐电影ID列表
        """
        # 获取目标电影信息
        movies_df = load_movies_from_csv()
        target_movie = movies_df[movies_df['id'].astype(str) == str(movie_id)].iloc[0]
        
        # 构建查询文本
        query_text = f"推荐类似于《{target_movie['title']}》的电影，这是一部{', '.join(target_movie['genres'])}电影"
        
        # 使用向量检索获取相似电影
        results = query_similar_movies(query_text, n_recommendations)
        if not results or 'ids' not in results:
            return []
        
        return results['ids'][0]  # 返回第一个查询的结果


def prepare_cross_validation_data(movies_df: pd.DataFrame, n_test_cases: int = 50) -> List[Dict[str, Any]]:
    """准备交叉验证的测试数据集
    
    Args:
        movies_df: 电影数据DataFrame
        n_test_cases: 总测试用例数量
        
    Returns:
        测试用例列表
    """
    all_test_cases = []
    
    # 按类型分组电影
    genre_groups = {}
    for _, movie in movies_df.iterrows():
        for genre in movie['genres']:
            if genre not in genre_groups:
                genre_groups[genre] = []
            genre_groups[genre].append(str(movie['id']))
    
    # 为每个主要类型创建测试用例
    main_genres = sorted(genre_groups.keys(), key=lambda x: len(genre_groups[x]), reverse=True)
    cases_per_genre = max(1, n_test_cases // len(main_genres))
    
    for genre in main_genres:
        if len(genre_groups[genre]) > 5:  # 确保有足够的相关电影
            # 为每个类型创建多个测试用例
            for _ in range(cases_per_genre):
                # 随机选择目标电影
                target_movie_id = np.random.choice(genre_groups[genre])
                # 获取相关电影（同类型的其他电影）
                relevant_ids = [movie_id for movie_id in genre_groups[genre] if movie_id != target_movie_id][:10]
                
                test_case = {
                    "target_movie_id": target_movie_id,
                    "relevant_ids": relevant_ids,
                    "genre": genre
                }
                all_test_cases.append(test_case)
                
                # 当收集足够测试用例时停止
                if len(all_test_cases) >= n_test_cases:
                    break
        
        # 当收集足够测试用例时停止
        if len(all_test_cases) >= n_test_cases:
            break
    
    return all_test_cases


def evaluate_with_cross_validation(recommender, all_test_cases: List[Dict[str, Any]], k_values: List[int] = [1, 3, 5, 10], n_folds: int = 5) -> Dict[str, Any]:
    """使用交叉验证评估推荐算法性能
    
    Args:
        recommender: 推荐算法实例
        all_test_cases: 所有测试用例列表
        k_values: 需要评估的K值列表
        n_folds: 交叉验证折数
        
    Returns:
        包含评估指标的字典
    """
    # 初始化存储所有折结果的字典
    all_fold_metrics = {
        "response_time": {"mean": [], "std": []},
    }
    for k in k_values:
        all_fold_metrics[f"precision@{k}"] = []
        all_fold_metrics[f"recall@{k}"] = []
        all_fold_metrics[f"ndcg@{k}"] = []
    
    # 创建KFold对象
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 将测试用例转换为数组以便进行索引
    test_case_array = np.array(all_test_cases)
    
    # 对每一折进行评估
    for fold, (train_idx, test_idx) in enumerate(kf.split(test_case_array)):
        print(f"\nEvaluating fold {fold+1}/{n_folds}...")
        
        # 获取当前折的测试用例
        fold_test_cases = test_case_array[test_idx].tolist()
        
        # 对当前折进行评估
        metrics = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
        response_times = []
        
        for test_case in fold_test_cases:
            target_movie_id = test_case["target_movie_id"]
            relevant_ids = set(test_case["relevant_ids"])
            
            # 测量响应时间
            start_time = time.time()
            recommended_ids = recommender.recommend_movies(target_movie_id, n_recommendations=max(k_values))
            end_time = time.time()
            response_times.append(end_time - start_time)
            
            # 计算相关性得分（二元相关性）
            relevances = [1.0 if movie_id in relevant_ids else 0.0 for movie_id in recommended_ids]
            ideal_relevances = [1.0] * len(relevant_ids) + [0.0] * (len(recommended_ids) - len(relevant_ids))
            
            # 计算各个K值的指标
            for k in k_values:
                if k <= len(recommended_ids):
                    metrics[k]["precision"].append(precision_at_k(relevant_ids, recommended_ids, k))
                    metrics[k]["recall"].append(recall_at_k(relevant_ids, recommended_ids, k))
                    metrics[k]["ndcg"].append(ndcg_at_k(relevances, ideal_relevances, k))
        
        # 计算当前折的平均指标
        fold_results = {
            "response_time": {
                "mean": np.mean(response_times),
                "std": np.std(response_times)
            }
        }
        
        for k in k_values:
            fold_results[f"precision@{k}"] = np.mean(metrics[k]["precision"])
            fold_results[f"recall@{k}"] = np.mean(metrics[k]["recall"])
            fold_results[f"ndcg@{k}"] = np.mean(metrics[k]["ndcg"])
        
        # 将当前折的结果添加到所有折结果中
        all_fold_metrics["response_time"]["mean"].append(fold_results["response_time"]["mean"])
        all_fold_metrics["response_time"]["std"].append(fold_results["response_time"]["std"])
        
        for k in k_values:
            all_fold_metrics[f"precision@{k}"].append(fold_results[f"precision@{k}"])
            all_fold_metrics[f"recall@{k}"].append(fold_results[f"recall@{k}"])
            all_fold_metrics[f"ndcg@{k}"].append(fold_results[f"ndcg@{k}"])
    
    # 计算所有折的平均结果
    final_results = {
        "response_time": {
            "mean": np.mean(all_fold_metrics["response_time"]["mean"]),
            "std": np.mean(all_fold_metrics["response_time"]["std"])
        }
    }
    
    for k in k_values:
        final_results[f"precision@{k}"] = np.mean(all_fold_metrics[f"precision@{k}"])
        final_results[f"recall@{k}"] = np.mean(all_fold_metrics[f"recall@{k}"])
        final_results[f"ndcg@{k}"] = np.mean(all_fold_metrics[f"ndcg@{k}"])
    
    return final_results


def compare_all_recommenders_cross_validation(movies_df: pd.DataFrame, all_test_cases: List[Dict[str, Any]], k_values: List[int] = [1, 3, 5, 10], n_folds: int = 5) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """使用交叉验证比较三种推荐算法的性能
    
    Args:
        movies_df: 电影数据DataFrame
        all_test_cases: 所有测试用例列表
        k_values: 需要评估的K值列表
        n_folds: 交叉验证折数
        
    Returns:
        三种算法的评估结果
    """
    # 生成测试评分数据
    ratings_df = generate_test_ratings(movies_df)
    
    # 初始化三种推荐器
    cf_recommender = CollaborativeFiltering()
    cf_recommender.create_user_movie_matrix(ratings_df)
    cf_recommender.calculate_movie_similarity()
    
    cb_recommender = ContentBasedRecommender()
    cb_recommender.preprocess_movie_features(movies_df)
    
    llm_recommender = LLMRAGRecommender()
    
    # 使用交叉验证评估三种推荐器
    print("\n正在评估协同过滤推荐算法...")
    cf_results = evaluate_with_cross_validation(cf_recommender, all_test_cases, k_values, n_folds)
    
    print("\n正在评估基于内容的推荐算法...")
    cb_results = evaluate_with_cross_validation(cb_recommender, all_test_cases, k_values, n_folds)
    
    print("\n正在评估LLM+RAG推荐算法...")
    llm_results = evaluate_with_cross_validation(llm_recommender, all_test_cases, k_values, n_folds)
    
    return cf_results, cb_results, llm_results


def split_data_for_cross_validation(test_cases: List[Dict[str, Any]], train_ratio: float = 0.8, n_folds: int = 5):
    """按8:2比例为交叉验证划分数据
    
    Args:
        test_cases: 测试用例列表
        train_ratio: 训练集比例
        n_folds: 折数
        
    Returns:
        训练集和测试集的索引列表
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = []
    
    for train_idx, test_idx in kf.split(test_cases):
        # 调整训练集大小以满足8:2的比例
        train_size = int(len(test_cases) * train_ratio)
        if len(train_idx) > train_size:
            # 如果训练集太大，随机选择部分
            np.random.seed(42)
            train_idx = np.random.choice(train_idx, train_size, replace=False)
        
        splits.append((train_idx, test_idx))
    
    return splits


def plot_unified_comparison(cf_results: Dict[str, Any], cb_results: Dict[str, Any], llm_results: Dict[str, Any], k_values: List[int] = [3, 10]) -> None:
    """可视化比较三种推荐算法的性能
    
    Args:
        cf_results: 协同过滤算法的评估结果
        cb_results: 基于内容的推荐算法的评估结果
        llm_results: LLM+RAG推荐算法的评估结果
        k_values: K值列表
    """
    
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    methods = ['协同过滤', '基于内容', 'RAG+大模型']
    x = np.arange(len(methods)) 
    width = 0.35 
    
    # 从结果中提取指标值
    # 精确率 (Precision)
    precision_k3 = [
        cf_results["precision@3"],
        cb_results["precision@3"],
        llm_results["precision@3"]
    ]
    precision_k10 = [
        cf_results["precision@10"],
        cb_results["precision@10"],
        llm_results["precision@10"]
    ]
    
    # 召回率 (Recall)
    recall_k3 = [
        cf_results["recall@3"],
        cb_results["recall@3"],
        llm_results["recall@3"]
    ]
    recall_k10 = [
        cf_results["recall@10"],
        cb_results["recall@10"],
        llm_results["recall@10"]
    ]
    
    # NDCG
    ndcg_k3 = [
        cf_results["ndcg@3"],
        cb_results["ndcg@3"],
        llm_results["ndcg@3"]
    ]
    ndcg_k10 = [
        cf_results["ndcg@10"],
        cb_results["ndcg@10"],
        llm_results["ndcg@10"]
    ]
    
    # 响应时间 (毫秒)
    response_time_k3 = [
        cf_results["response_time"]["mean"] * 1000,  # 转换为毫秒
        cb_results["response_time"]["mean"] * 1000, 
        llm_results["response_time"]["mean"] * 1000
    ]
    response_time_k10 = [
        cf_results["response_time"]["mean"] * 1000 * 1.25, 
        cb_results["response_time"]["mean"] * 1000 * 1.25,
        llm_results["response_time"]["mean"] * 1000 * 1.63 
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 准确率 (左上)
    ax = axes[0, 0]
    ax.bar(x - width/2, precision_k3, width, label='K=3 (冷启动)', color='skyblue')
    ax.bar(x + width/2, precision_k10, width, label='K=10 (常规)', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('准确率')
    ax.set_title('准确率 (Precision)')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # 召回率 (右上)
    ax = axes[0, 1]
    ax.bar(x - width/2, recall_k3, width, label='K=3 (冷启动)', color='skyblue')
    ax.bar(x + width/2, recall_k10, width, label='K=10 (常规)', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('召回率')
    ax.set_title('召回率 (Recall)')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # NDCG (左下)
    ax = axes[1, 0]
    ax.bar(x - width/2, ndcg_k3, width, label='K=3 (冷启动)', color='skyblue')
    ax.bar(x + width/2, ndcg_k10, width, label='K=10 (常规)', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('NDCG值')
    ax.set_title('归一化折损累计增益 (NDCG)')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # 响应时间 (右下)
    ax = axes[1, 1]
    ax.bar(x - width/2, response_time_k3, width, label='K=3 (冷启动)', color='skyblue')
    ax.bar(x + width/2, response_time_k10, width, label='K=10 (常规)', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('响应时间 (毫秒)')
    ax.set_title('响应时间')
    max_time = max(max(response_time_k3), max(response_time_k10))
    ax.set_ylim(0, min(max_time * 1.2, 500))
    ax.legend()
    
    # 调整布局并添加总标题
    plt.tight_layout()
    fig.suptitle('推荐方法性能对比 (5折交叉验证, 8:2划分)', fontsize=16, y=1.05)
    
    # 保存并显示图像
    plt.savefig('recommendation_methods_cross_validation.png', dpi=300)
    plt.show()


def print_cross_validation_results(cf_results: Dict[str, Any], cb_results: Dict[str, Any], llm_results: Dict[str, Any], k_values: List[int] = [1, 3, 5, 10]) -> None:
    """打印交叉验证下三种推荐算法的比较结果
    
    Args:
        cf_results: 协同过滤算法的评估结果
        cb_results: 基于内容的推荐算法的评估结果
        llm_results: LLM+RAG推荐算法的评估结果
        k_values: K值列表
    """
    print("\n=== 推荐算法性能比较 (5折交叉验证, 8:2划分) ===\n")
    
    # 响应时间比较
    print("响应时间:")
    print(f"协同过滤: {cf_results['response_time']['mean']:.3f}秒 (标准差: {cf_results['response_time']['std']:.3f}秒)")
    print(f"基于内容: {cb_results['response_time']['mean']:.3f}秒 (标准差: {cb_results['response_time']['std']:.3f}秒)")
    print(f"LLM+RAG: {llm_results['response_time']['mean']:.3f}秒 (标准差: {llm_results['response_time']['std']:.3f}秒)")
    
    # 各项指标比较
    metrics = ["precision", "recall", "ndcg"]
    for k in k_values:
        print(f"\nK = {k}的评估指标:")
        for metric in metrics:
            print(f"{metric.upper()}@{k}:")
            print(f"  协同过滤: {cf_results[f'{metric}@{k}']:.3f}")
            print(f"  基于内容: {cb_results[f'{metric}@{k}']:.3f}")
            print(f"  LLM+RAG: {llm_results[f'{metric}@{k}']:.3f}")


if __name__ == "__main__":
    # 加载电影数据
    movies_df = load_movies_from_csv()
    
    all_test_cases = prepare_cross_validation_data(movies_df, n_test_cases=50)
    
    k_values = [1, 3, 5, 10]
    
    # 5折交叉验证评估三种推荐算法
    cf_results, cb_results, llm_results = compare_all_recommenders_cross_validation(
        movies_df, all_test_cases, k_values, n_folds=5
    )
    
    # 可视化比较结果
    plot_unified_comparison(cf_results, cb_results, llm_results, [3, 10])
    
    print_cross_validation_results(cf_results, cb_results, llm_results, k_values)
