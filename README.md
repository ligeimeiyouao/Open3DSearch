<div align="center"><h1>Open3DSearch: Zero-Shot Precise Retrieval of 3D Shapes Using Text Descriptions</h1></div>
This repository is the official implementation of <a href="https://dl.acm.org/doi/10.1145/3746027.3755533">Open3DSearch</a>🔥🔥🔥
<div>
  <h3>Introduction</h3>
  <p>With the rapid growth of 3D content, there is an increasing need for intelligent systems that can search for complex 3D shapes using simple natural language queries. However, existing approaches face significant limitations. They rely heavily on manually labeled datasets and use fixed similarity thresholds to determine matches, which restricts their ability to generalize and accurately retrieve novel or diverse 3D shapes. To bridge these gaps, this paper introduces Open3DSearch, the first attempt to address the challenge of open-domain text-to-shape precise retrieval. Our core idea is to transform 3D shapes into semantically representative 2D views, thereby enabling the task to be handled by mature large vision-language models (LVLMs) and allowing for explicit cross-modal matching judgments. To realize this concept, we design a view rendering strategy to mitigate potential information degradation during 3D-to-2D conversion while capturing the maximal amount of query-relevant information. To evaluate Open3DSearch and advance research in this field, we present the Uni3D-R benchmark dataset, designed to simulate precise associations between user queries and 3D shapes in open-domain contexts. Extensive quantitative and qualitative experiments demonstrate that Open3DSearch achieves state-of-the-art results.</p>
</div>

  <h2>Installation</h2>
  <p>Create a conda environment and install basic dependencies:</p>

  ```bash
  conda create -n Open3DSearch python=3.9
  conda activate Open3DSearch

  # Install the according versions of torch and torchvision
  pip install torch==2.2.2+cu118 torchaudio==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

  # Install required dependencies
  pip install -r requirements.txt
  ```  
 <p>Some libraries may need to be installed according to your specific device configuration (e.g., <a href="https://www.dgl.ai/pages/start.html">dgl</a>).</p>
 <h2>DataSets & data preprocessing</h2>
 <p>We propose and employ the Uni3D-R dataset, which contains 7,855 3D shapes and 812 carefully curated query texts.<br>
  You can download it from <a href="https://huggingface.co/datasets/ligeimeiyouao/Uni3D-R">here</a>, place 3dmodel_query_matches.json(containing 812 query statements and their corresponding 3D shapes) together with the npys files into <code style="background: #eee; color: #333;">/data/</code>, and put download_glbs.py into <code style="background: #eee; color: #333;">/data_preprocessing/</code>.
 </p>
 <p>After downloading the dataset, you need to run<a href="https://github.com/ligeimeiyouao/Open3DSearch/blob/main/data_preprocessing/download_glbs.py">download_glbs.py</a>to download the GLB files (by default, they are downloaded to the .objaverse folder in the root directory of the device). In addition, we also provide the 3D shape feature extraction code<a href="https://github.com/ligeimeiyouao/Open3DSearch/blob/main/data_preprocessing/3d_shape_feature_extraction.py">3d_shape_feature_extraction.py</a></p>
<h2>Quickstart</h2>
<p>
  After data preprocessing, you can run <a href="https://github.com/ligeimeiyouao/Open3DSearch/blob/main/main/demo.py"><code style="background: #eee; color: #333;">/main/demo.py</code></a> to test the 3D model retrieval results for custom queries.If you need to predict matching results for multiple queries, please refer to <a href="https://github.com/ligeimeiyouao/Open3DSearch/blob/main/main/run.py"><code style="background: #eee; color: #333;">/main/run.py</code></a>.
</p>

```bash
python main/demo.py
```
