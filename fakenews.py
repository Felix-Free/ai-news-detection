from flask import Flask, render_template, request, jsonify
import os
import jieba
import re
from collections import Counter

app = Flask(__name__)

# 加载专业术语词典（示例）
professional_terms = set(['人工智能', '机器学习', '深度学习', '神经网络', '大数据', '云计算', '区块链'])

def analyze_text(text):
    # 分割标题和正文（假设第一行是标题）
    lines = text.strip().split('\n')
    title = lines[0] if lines else ''
    content = '\n'.join(lines[1:]) if len(lines) > 1 else ''
    
    # 分词
    title_tokens = list(jieba.cut(title))
    content_tokens = list(jieba.cut(content))
    
    # 计算专业术语
    title_terms = [word for word in title_tokens if word in professional_terms]
    content_terms = [word for word in content_tokens if word in professional_terms]
    
    # 情感分析（简单示例）
    positive_words = ['好', '优秀', '成功', '进步', '发展']
    negative_words = ['坏', '失败', '问题', '困难', '挑战']
    
    content_words = content_tokens
    positive_count = sum(1 for word in content_words if word in positive_words)
    negative_count = sum(1 for word in content_words if word in negative_words)
    
    return {
        'title_length': len(title_tokens),
        'content_length': len(content_tokens),
        'term_density': len(content_terms) / len(content_tokens) if content_tokens else 0,
        'term_error_rate': 0.1,  # 示例值，实际需要更复杂的计算
        'sentiment_ratio': (positive_count + negative_count) / len(content_tokens) if content_tokens else 0,
        'sentiment_bias': (positive_count - negative_count) / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    text = request.form.get('text', '')
    # 获取文本特征
    features = analyze_text(text)
    # 这里可以添加实际的虚假新闻检测逻辑
    # 目前返回一个模拟的结果
    result = {
        'probability': 75.5,
        'is_fake': True,
        'features': features
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')

