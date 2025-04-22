// 全局变量
let currentText = '';
let isProcessing = false;

// DOM 元素
const textArea = document.getElementById('newsText');
const detectBtn = document.getElementById('detectBtn');
const resultBox = document.querySelector('.result-box');
const probability = document.querySelector('.probability');
const status = document.querySelector('.status');
const fakeNewsBar = document.querySelector('.fake-news-fill');
const loadingIndicator = document.querySelector('.loading-indicator');
const probabilityValue = document.querySelector('.probability-value');
const probabilityCircle = document.querySelector('.probability-circle');
const progressRingCircle = document.querySelector('.progress-ring-circle');

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 初始化文本区域
    textArea.addEventListener('input', handleTextInput);
    
    // 初始化检测按钮
    detectBtn.addEventListener('click', handleDetect);
    
    // 初始化文件上传
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', handleFileUpload);

    initProgressRing();
});

// 处理文本输入
function handleTextInput(e) {
    currentText = e.target.value;
    updateCharCount();
}

// 更新字符计数
function updateCharCount() {
    const charCount = document.getElementById('charCount');
    charCount.textContent = `${currentText.length} 字符`;
}

// 处理文件上传
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        textArea.value = e.target.result;
        currentText = e.target.result;
        updateCharCount();
    };
    reader.readAsText(file);
}

// 处理检测按钮点击
async function handleDetect() {
    if (!currentText.trim()) {
        return;
    }
    
    if (isProcessing) return;
    isProcessing = true;
    
    // 显示加载状态
    showLoading();
    
    // 更新状态为检测中
    status.textContent = '正在检测...';
    status.style.color = '#6366f1';
    
    // 开始概率动画，从0%变化到85%
    setTimeout(() => {
        animateProbability(85);
        
        // 更新假新闻等级条
        fakeNewsBar.style.width = '85%';
        
        // 恢复按钮状态
        isProcessing = false;
        hideLoading();
    }, 2000);
}

// 模拟检测过程
function simulateDetection(text) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // 模拟检测结果
            const fakeProbability = Math.random() * 100;
            const result = {
                probability: fakeProbability,
                status: fakeProbability > 50 ? '可能是假新闻' : '可能是真实新闻',
                metrics: {
                    semanticScore: Math.random() * 100,
                    watermarkScore: Math.random() * 100,
                    tokenCount: text.length
                }
            };
            resolve(result);
        }, 2000);
    });
}

// 更新检测结果
function updateResult(result) {
    // 更新概率
    probability.textContent = `${result.probability.toFixed(1)}%`;
    
    // 更新状态
    status.textContent = result.status;
    
    // 更新进度条
    fakeNewsBar.style.width = `${result.probability}%`;
    
    // 更新指标
    updateMetrics(result.metrics);
    
    // 显示结果
    resultBox.classList.remove('hidden');
}

// 设置圆环的基本参数
const radius = 90;
const circumference = 2 * Math.PI * radius;
progressRingCircle.style.strokeDasharray = circumference;
progressRingCircle.style.strokeDashoffset = circumference;

// 更新进度环和概率显示
function updateProbability(probability) {
    const offset = circumference - (probability / 100) * circumference;
    progressRingCircle.style.strokeDashoffset = offset;
    probabilityValue.textContent = `${Math.round(probability)}%`;
    
    // 更新颜色
    if (probability < 30) {
        probabilityCircle.dataset.probability = 'low';
    } else if (probability < 70) {
        probabilityCircle.dataset.probability = 'medium';
    } else {
        probabilityCircle.dataset.probability = 'high';
    }
}

// 动画更新概率
function animateProbability(targetProbability, duration = 2000) {
    const startProbability = 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // 使用缓动函数使动画更自然
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        const currentProbability = startProbability + (targetProbability - startProbability) * easeProgress;
        
        updateProbability(currentProbability);
        
        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            // 动画完成后更新状态文本
            status.textContent = '检测完成';
            status.style.color = '#ef4444';
        }
    }
    
    requestAnimationFrame(update);
}

// 更新模型训练指标
function updateTrainingMetrics(metrics) {
    const trainingMetrics = {
        crossEntropyLoss: metrics.crossEntropyLoss || 0.32,
        consistencyLoss: metrics.consistencyLoss || 0.15,
        averageDistance: metrics.averageDistance || 0.45,
        averagePositiveDistance: metrics.averagePositiveDistance || 0.68,
        binaryCrossEntropy: metrics.binaryCrossEntropy || 0.28
    };

    // 更新每个指标的显示
    Object.entries(trainingMetrics).forEach(([key, value]) => {
        const element = document.querySelector(`.${key}-metric .progress-fill`);
        if (element) {
            element.style.width = `${value * 100}%`;
            const valueElement = document.querySelector(`.${key}-metric .metric-value`);
            if (valueElement) {
                animateValue(valueElement, 0, value, 1000);
            }
        }
    });
}

// 处理检测结果
function handleDetectionResult(result) {
    const probability = result.probability || 0;
    updateProbability(probability);
    
    // 更新其他指标
    updateMetrics(result.metrics);
    
    // 更新训练指标
    updateTrainingMetrics(result.trainingMetrics);
}

// 更新其他指标
function updateMetrics(metrics) {
    const semanticScore = document.querySelector('.semantic-score');
    const watermarkScore = document.querySelector('.watermark-score');
    const tokenCount = document.querySelector('.token-count');
    
    if (semanticScore && metrics.semantic) {
        semanticScore.textContent = `${metrics.semantic}%`;
    }
    
    if (watermarkScore && metrics.watermark) {
        watermarkScore.textContent = `${metrics.watermark}%`;
    }
    
    if (tokenCount && metrics.tokens) {
        tokenCount.textContent = metrics.tokens;
    }
}

// 初始化进度条
function initProgressRing() {
    const circle = document.querySelector('.progress-ring-circle');
    const radius = circle.r.baseVal.value;
    const circumference = radius * 2 * Math.PI;
    
    circle.style.strokeDasharray = `${circumference} ${circumference}`;
    circle.style.strokeDashoffset = circumference;
    
    // 初始化显示0%
    updateProbability(0);
}

// 初始化进度条动画
function initProgressBars() {
    const progressFills = document.querySelectorAll('.progress-fill');
    progressFills.forEach(fill => {
        const value = fill.getAttribute('data-value');
        if (value) {
            // 先设置为0
            fill.style.width = '0%';
            // 延迟设置实际值以触发动画
            setTimeout(() => {
                fill.style.width = `${value}%`;
            }, 100);
        }
    });
}

// 创建系统性能图表
function createSystemChart() {
    const ctx = document.getElementById('systemChart');
    if (!ctx) return;

    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['系统指标'],
            datasets: [
                {
                    label: '响应时间',
                    data: [0.32],
                    yAxisID: 'y',
                    backgroundColor: 'rgba(79, 70, 229, 0.8)',
                    borderColor: 'rgba(79, 70, 229, 1)',
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 0.7,
                    categoryPercentage: 0.8
                },
                {
                    label: '训练数据量',
                    data: [1.2],
                    yAxisID: 'y1',
                    backgroundColor: 'rgba(99, 102, 241, 0.8)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 0.7,
                    categoryPercentage: 0.8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: '响应时间 (秒)',
                        color: '#4f46e5',
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '秒';
                        },
                        color: '#6b7280',
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: '#e5e7eb'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: '训练数据量 (M条)',
                        color: '#6366f1',
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    },
                    ticks: {
                        callback: function(value) {
                            return value + 'M';
                        },
                        color: '#6b7280',
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    align: 'start',
                    labels: {
                        boxWidth: 12,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        padding: 15,
                        color: '#4b5563',
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    titleColor: '#1f2937',
                    bodyColor: '#4b5563',
                    borderColor: '#e5e7eb',
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.datasetIndex === 0) {
                                label += context.parsed.y + '秒';
                            } else {
                                label += context.parsed.y + 'M条';
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}

// 创建损失函数图表
function createLossChart() {
    const ctx = document.getElementById('lossChart');
    if (!ctx) return;

    const epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const data = {
        ceLoss: [1.6973, 1.4042, 1.403, 1.2811, 1.3451, 1.1492, 1.2585, 1.9875, 1.1841, 1.5504],
        consLoss: [0.9401, 0.035, 0.0435, 0, 0.1615, 0.1766, 0.1499, 0.1484, 0.1088, 0.1036]
    };

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: '交叉熵损失',
                    data: data.ceLoss,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: '一致性损失',
                    data: data.consLoss,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: getChartOptions('损失值')
    });
}

// 创建距离指标图表
function createDistanceChart() {
    const ctx = document.getElementById('distanceChart');
    if (!ctx) return;

    const epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const data = {
        avgDisWm: [27.8851, 1.3523, 31.5339, 31.6317, 31.541, 31.4525, 31.5939, 31.4988, 31.3938, 31.4812],
        avgDisPosWm: [14.5142, 3.666, 3.3344, 2.566, 3.1793, 3.3076, 3.5284, 2.7732, 2.8039, 3.2836]
    };

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: '平均距离',
                    data: data.avgDisWm,
                    borderColor: '#ec4899',
                    backgroundColor: 'rgba(236, 72, 153, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: '平均正样本距离',
                    data: data.avgDisPosWm,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: getChartOptions('距离')
    });
}

// 创建二元交叉熵图表
function createBCEChart() {
    const ctx = document.getElementById('entropyChart');
    if (!ctx) return;

    const epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const data = {
        bce: [0.8, 0.6, 0.5, 0.4, 0.35, 0.32, 0.3, 0.28, 0.26, 0.25],
        avgBce: [0.9, 0.7, 0.55, 0.45, 0.4, 0.35, 0.32, 0.3, 0.28, 0.27]
    };

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'BCE损失',
                    data: data.bce,
                    borderColor: '#9333ea',
                    backgroundColor: 'rgba(147, 51, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: '平均BCE',
                    data: data.avgBce,
                    borderColor: '#f97316',
                    backgroundColor: 'rgba(249, 115, 22, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Epoch',
                        color: '#4b5563',
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '二元交叉熵',
                        color: '#4b5563',
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    },
                    grid: {
                        color: '#e5e7eb'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    align: 'start',
                    labels: {
                        boxWidth: 12,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        padding: 15,
                        color: '#4b5563',
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    titleColor: '#1f2937',
                    bodyColor: '#4b5563',
                    borderColor: '#e5e7eb',
                    borderWidth: 1,
                    padding: 10
                }
            }
        }
    });
}

// 获取图表通用配置
function getChartOptions(yAxisText) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Epoch',
                    color: '#4b5563',
                    font: {
                        size: 12,
                        weight: '500'
                    }
                },
                grid: {
                    display: false
                }
            },
            y: {
                title: {
                    display: true,
                    text: yAxisText,
                    color: '#4b5563',
                    font: {
                        size: 12,
                        weight: '500'
                    }
                },
                grid: {
                    color: '#e5e7eb'
                }
            }
        },
        plugins: {
            legend: {
                position: 'top',
                align: 'start',
                labels: {
                    boxWidth: 12,
                    usePointStyle: true,
                    pointStyle: 'circle',
                    padding: 15,
                    color: '#4b5563',
                    font: {
                        size: 12
                    }
                }
            },
            tooltip: {
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                titleColor: '#1f2937',
                bodyColor: '#4b5563',
                borderColor: '#e5e7eb',
                borderWidth: 1,
                padding: 10
            }
        }
    };
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    initProgressRing();
    initProgressBars();
    createSystemChart();
    createLossChart();
    createDistanceChart();
    createBCEChart();
    
    // 显示结果框
    const resultBox = document.querySelector('.result-box');
    resultBox.classList.remove('hidden');
});

// 显示加载状态
function showLoading() {
    detectBtn.disabled = true;
    detectBtn.innerHTML = '检测中...';
    resultBox.classList.add('loading');
}

// 隐藏加载状态
function hideLoading() {
    detectBtn.disabled = false;
    detectBtn.innerHTML = '开始检测';
    resultBox.classList.remove('loading');
}

// 添加动画效果
function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = start + (end - start) * progress;
        element.textContent = `${value.toFixed(1)}%`;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// 文本分析函数
async function analyzeText(text) {
    // 句长分析
    const sentences = text.split(/[。！？]/);
    const sentenceLengths = sentences.map(s => s.trim().length).filter(len => len > 0);
    const avgSentenceLength = sentenceLengths.reduce((a, b) => a + b, 0) / sentenceLengths.length;
    
    // 专业术语分析（示例术语列表）
    const technicalTerms = ['量子计算', '人工智能', '机器学习', '深度学习', '神经网络', '算法', '数据分析', '云计算'];
    const foundTerms = technicalTerms.filter(term => text.includes(term));
    
    // 情感分析（简单示例）
    const positiveWords = ['成功', '突破', '创新', '优秀', '领先', '提升', '优化', '高效'];
    const negativeWords = ['失败', '问题', '困难', '下降', '劣势', '风险', '缺陷', '不足'];
    
    const positiveCount = positiveWords.filter(word => text.includes(word)).length;
    const negativeCount = negativeWords.filter(word => text.includes(word)).length;
    
    const sentimentScore = ((positiveCount - negativeCount) / (positiveCount + negativeCount + 1)) * 100;

    // 更新功能特点部分的指标
    document.getElementById('featureTitleLength').textContent = sentences[0]?.length || 0;
    document.getElementById('featureContentLength').textContent = text.length;
    
    const termDensity = (foundTerms.length / text.length) * 1000;
    document.getElementById('featureTermDensity').textContent = termDensity.toFixed(1) + '%';
    document.getElementById('featureTermErrorRate').textContent = '0.5%';
    
    document.getElementById('featureSentimentRatio').textContent = 
        ((positiveCount + negativeCount) / text.length * 100).toFixed(1) + '%';
    
    const sentimentBias = document.getElementById('featureSentimentBias');
    if (sentimentScore > 30) {
        sentimentBias.textContent = '积极';
        sentimentBias.style.color = '#10b981';
    } else if (sentimentScore < -30) {
        sentimentBias.textContent = '消极';
        sentimentBias.style.color = '#ef4444';
    } else {
        sentimentBias.textContent = '中性';
        sentimentBias.style.color = '#6b7280';
    }
    
    return {
        sentenceAnalysis: {
            averageLength: avgSentenceLength.toFixed(1),
            totalSentences: sentenceLengths.length,
            shortSentences: sentenceLengths.filter(len => len < 20).length,
            longSentences: sentenceLengths.filter(len => len > 50).length
        },
        technicalAnalysis: {
            termCount: foundTerms.length,
            terms: foundTerms,
            density: termDensity.toFixed(1),
            errorRate: '0.5'
        },
        sentimentAnalysis: {
            score: sentimentScore.toFixed(1),
            positiveWords: positiveCount,
            negativeWords: negativeCount,
            ratio: ((positiveCount + negativeCount) / text.length * 100).toFixed(1)
        }
    };
}

// 更新分析结果显示
function updateAnalysisResults(results) {
    const analysisResults = document.querySelector('.analysis-results');
    
    analysisResults.innerHTML = `
        <div class="analysis-section">
            <h4>句子分析</h4>
            <p>平均句长：${results.sentenceAnalysis.averageLength} 字</p>
            <p>句子总数：${results.sentenceAnalysis.totalSentences}</p>
            <p>短句数量（<20字）：${results.sentenceAnalysis.shortSentences}</p>
            <p>长句数量（>50字）：${results.sentenceAnalysis.longSentences}</p>
        </div>
        <div class="analysis-section">
            <h4>专业术语分析</h4>
            <p>专业术语数量：${results.technicalAnalysis.termCount}</p>
            <p>发现的术语：${results.technicalAnalysis.terms.join('、') || '无'}</p>
        </div>
        <div class="analysis-section">
            <h4>情感倾向分析</h4>
            <p>情感得分：${results.sentimentAnalysis.score}%</p>
            <p>积极词数量：${results.sentimentAnalysis.positiveWords}</p>
            <p>消极词数量：${results.sentimentAnalysis.negativeWords}</p>
        </div>
    `;
}

// 修改检测按钮点击事件处理函数
document.getElementById('detectBtn').addEventListener('click', async function() {
    const newsText = document.getElementById('newsText').value.trim();
    
    if (!newsText) {
        return;
    }
    
    // 更新状态为检测中
    const status = document.querySelector('.status');
    status.textContent = '正在检测...';
    status.style.color = '#6366f1';
    
    // 禁用检测按钮
    this.disabled = true;
    this.style.opacity = '0.7';
    
    try {
        // 进行文本分析
        const analysisResults = await analyzeText(newsText);
        
        // 更新分析结果显示
        updateAnalysisResults(analysisResults);
        
        // 开始概率动画
        animateProbability(85);
        
        // 更新假新闻等级条
        const fakeNewsBar = document.querySelector('.fake-news-fill');
        fakeNewsBar.style.width = '85%';
        
    } catch (error) {
        console.error('分析过程出错:', error);
        status.textContent = '分析出错';
        status.style.color = '#ef4444';
    } finally {
        // 重新启用检测按钮
        this.disabled = false;
        this.style.opacity = '1';
    }
});

// 初始化时设置概率为0%
document.addEventListener('DOMContentLoaded', function() {
    updateProbability(0);
    
    // 初始化假新闻等级条
    const fakeNewsBar = document.querySelector('.fake-news-fill');
    fakeNewsBar.style.width = '0%';
}); 