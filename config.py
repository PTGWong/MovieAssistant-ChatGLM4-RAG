# 配置文件，存储API密钥和其他配置信息

# API密钥
TMDB_API_KEY = 'your_tmdb_api_key_here'
ZHIPU_API_KEY = 'your_zhipu_api_key_here'
SILICONFLOW_API_KEY = 'your_siliconflow_api_key_here'

# API URL
SILICONFLOW_API_URL = 'https://api.siliconflow.cn/v1/chat/completions'

# 向量模型配置
EMBEDDING_MODEL = 'Pro/BAAI/bge-m3'

# LLM模型配置
LLM_MODEL = 'glm-4'

# 向量数据库配置
CHROMA_COLLECTION_NAME = 'movies'
CHROMA_PERSIST_DIRECTORY = './chroma_db'

# 其他配置
MAX_RECOMMENDATIONS = 5