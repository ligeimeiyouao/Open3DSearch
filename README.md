<div align="center"><h2>Open3DSearch: Zero-Shot Precise Retrieval of 3D Shapes Using Text Descriptions</h2></div>
This repository is the official implementation of <a href="https://dl.acm.org/doi/10.1145/3746027.3755533">Open3DSearch</a>
<div>
  <h3>Introduction</h3>
  <p>With the rapid growth of 3D content, there is an increasing need for intelligent systems that can search for complex 3D shapes using simple natural language queries. However, existing approaches face significant limitations. They rely heavily on manually labeled datasets and use fixed similarity thresholds to determine matches, which restricts their ability to generalize and accurately retrieve novel or diverse 3D shapes. To bridge these gaps, this paper introduces Open3DSearch, the first attempt to address the challenge of open-domain text-to-shape precise retrieval. Our core idea is to transform 3D shapes into semantically representative 2D views, thereby enabling the task to be handled by mature large vision-language models (LVLMs) and allowing for explicit cross-modal matching judgments. To realize this concept, we design a view rendering strategy to mitigate potential information degradation during 3D-to-2D conversion while capturing the maximal amount of query-relevant information. To evaluate Open3DSearch and advance research in this field, we present the Uni3D-R benchmark dataset, designed to simulate precise associations between user queries and 3D shapes in open-domain contexts. Extensive quantitative and qualitative experiments demonstrate that Open3DSearch achieves state-of-the-art results.</p>
</div>
<h3>Structure overview</h3>
<div align="center"><img width="1300" height="400" alt="image" src="https://github.com/user-attachments/assets/28cdb7dc-33ba-4afb-b731-9bfb6974369d" />

</div>
