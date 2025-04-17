# 英文网页词频分析工具

本工具用于从多个给定的英文网页文章中提取文本内容，分析词语出现频率，并生成词云、词频热力图和词频评分表。

## 功能

- 从指定的 URL 列表获取网页文本内容。
- 对英文文本进行预处理：分词、转小写、去除标点符号、数字和常见停用词。
- 统计单词频率。
- 生成可视化结果：
    - 词云 (`word_cloud.png`)
    - 词频热力图 (`frequency_heatmap.png`)
    - 词频评分表 (`frequency_table.csv`)
- 结果保存在 `word_analysis_results` 文件夹中。

## 使用步骤

1.  **安装依赖库:**
    打开终端或命令提示符，导航到项目目录（包含 `requirements.txt` 文件的目录），然后运行以下命令安装所需的 Python 库：
    ```bash
    pip install -r requirements.txt
    ```
    首次运行时，NLTK 库可能需要下载额外的数据（停用词表和分词器）。脚本会自动尝试下载，请确保您的计算机可以访问互联网。

2.  **配置目标 URL:**
    打开 `word_analyzer.py` 文件，找到 `URLS` 列表。将列表中的示例 URL 替换为您想要分析的实际英文网页文章的 URL。
    ```python
    # --- Configuration ---
    # List of URLs to analyze
    URLS = [
        # 在这里添加你的目标 URL，例如:
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://www.example-news.com/article-on-technology"
        # 添加更多 URL...
    ]
    ```

3.  **运行脚本:**
    在终端或命令提示符中，确保您位于项目目录下，然后运行 Python 脚本：
    ```bash
    python word_analyzer.py
    ```

4.  **查看结果:**
    脚本执行完毕后，会在项目目录下创建一个名为 `word_analysis_results` 的文件夹。里面包含了生成的词云图片、热力图图片和词频 CSV 文件。
    同时，脚本也会在终端打印出分析过程和排名前 N 的词语及其频率。

## 注意

- 请确保提供的 URL 是有效的，并且可以公开访问。
- 网页内容的提取效果取决于网页的结构。对于结构复杂或使用大量 JavaScript 动态加载内容的网站，文本提取可能不完整。
- 脚本默认分析英文文本。对于其他语言，需要修改停用词列表和可能的分词逻辑。
- 可以在 `word_analyzer.py` 中调整 `OUTPUT_DIR` (输出目录名) 和 `TOP_N_WORDS` (热力图和表格中显示的词数量) 参数。