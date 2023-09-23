# Portofolio
---
## Data Science

### Thesis : Session-based Recommendation System using Gated Graph Neural Network and Attention Mechanism on E-commerce Dataset

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/luthfiraditya/Session-based-Recommendation-System){:target="_blank"}


<div style="text-align: justify">The primary objective of this research is to develop a session-based recommendation system that leverages a Gated Graph Neural Network (GGNN) to amalgamate global preferences, current interests, and user-specific preferences.
The research encompasses several stages, including data pre-processing, session graph construction, item embedding, session embedding, recommendation generation, and model evaluation.The research findings demonstrate that the proposed model exhibits superior performance compared to other models on both datasets. In the Yoochoose dataset, the Recall@20 achieved an impressive 70.81%, with MRR@20 reaching 31.05%. Meanwhile, the Diginetica dataset yielded a Recall@20 of 50.08% and an MRR@20 of 17.80%. These results signify the model's effectiveness in providing top-tier recommendations.

</div>
<br>
<center><img src="images/metode_penelitian.png"/></center>

---

### Named Entity Recognition (NER) using LayoutLM

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/luthfiraditya/LayoutLM-Document-Classification){:target="_blank"}


<div style="text-align: justify">
Working on a project that involves utilizing LayoutLM, a multimodal Transformer model designed for document image understanding and information extraction. This technology has promising applications in text comprehension and receipt understanding. Named Entity Recognition, a critical task, involves the identification and categorization of entities like individual names, organizations, locations, dates, and more within a given text. The innovative aspect of LayoutLM lies in its extension of this task to encompass the visual layout of text. This includes factors such as the positioning of text, font styles, and formatting details. Through my project, I am creating a system that can automatically extract and categorize named entities present in various documents. By combining the capabilities of language comprehension with visual layout analysis, I aim to achieve a more accurate identification of entities. This approach becomes particularly valuable in cases where conventional NER methods face challenges due to the diverse formats and structures of documents.


Original LayoutLM paper: https://arxiv.org/abs/1912.13318

LayoutLM docs in the Transformers library: https://huggingface.co/transformers/model_doc/layoutlm.html
</div>
<br>
<center><img src="images/schema.jpg"/></center>


---

---

### Document Classification using LayoutLM

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/luthfiraditya/LayoutLM-Document-Classification){:target="_blank"}


<div style="text-align: justify">
In this project, I am exploring document classification using LayoutLM, a pre-trained model for document image understanding. The goal of this project is to provide an overview of how to use LayoutLM for document classification, including how to prepare the data, fine-tune LayoutLMForSequenceClassification the pre-trained model, and evaluate its performance on RVL-CDIP dataset. By the end of this project, we should have a good understanding of how to use LayoutLM for document classification and be able to apply these techniques to document classification tasks.

Original LayoutLM paper: https://arxiv.org/abs/1912.13318

LayoutLM docs in the Transformers library: https://huggingface.co/transformers/model_doc/layoutlm.html
</div>
<br>
<center><img src="images/doc_class.png"/></center>


---

### Recommender Systems on E-commerce

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/luthfiraditya/Ecommerce-Recommendation-System){:target="_blank"}

<div style="text-align: justify">This project is a pilot project during internship. I built recommender systems for recommending products to user using Model-based recommendation system. The goal of this project is to make a recommendation system model that is more accurate than the previous model. <b>The model achieve the best performance with SVD++ where this model gets an RMSE score of 0.844 and MAE 0.384. I also use the mlflow tool to do experiment tracking.</b>
</div>
<br>
<center><img src="images/Recsys.png"/></center>

---
### Boston Housing Prediction with deployment

[![Open Notebook](https://img.shields.io/badge/Heroku-Open_Web_App-purple?logo=heroku)](projects/detect-food-trends-facebook.html){:target="_blank"}
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/luthfiraditya/Boston-Housing-Prediction-with-deployment){:target="_blank"}

<div style="text-align: justify">This project was started as a motivation for learning Machine Learning Algorithms and to learn the different data preprocessing techniques and <b>implement the concept of Homodescascity, Multicollinearity & Error terms distribution during data exploration. I also deploy this project in heroku. This model get R2 score of 73% and RMSE of 5.09</b>.
</div>
<center><img src="images/BostonHouse.png"></center>


---
### Location Recommendation for Retail

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/detect-spam-messages-nlp/blob/master/detect-spam-nlp.ipynb){:target="_blank"}
[![Open Research Poster](https://img.shields.io/badge/PDF-Open_Final_Report-red?logo=adobe-acrobat-reader&logoColor=red)](https://docs.google.com/document/d/1RF5hePterte23m-4obD89ax8giUqsfgE/edit?usp=sharing&ouid=109361563889763484237&rtpof=true&sd=true){:target="_blank"}


<div style="text-align: justify"><b>The purpose of this project is to provide location recommendations for retailers</b> who want to open offline stores. This project used to build the startup for the final project called Map.it and <b>succeeded in becoming the five best final projects during MBKM event</b>.</div>
<br>
<center><img src="images/locationintel.png"/></center>
<br>

---
### Herd Immunity Prediction

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/luthfiraditya/Herd-Immunity-Prediction){:target="_blank"}

<div style="text-align: justify"><b>This project aims to predict when Indonesia will reach 60-85% herd immunity</b> with COVID-19 vaccinations. Performing time series analysis and modeling with polynomial models. <b>Using degree=3 as the best degree that gets a score of r2=0.963.</b>
</div>
<br>
<center><img src="images/Herdimmunity.png"></center>
<br>

---
### Market Basket Analysis

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/luthfiraditya/Market-Basket-Analysis){:target="_blank"}

<div style="text-align: justify">The objective of this project is to <b>analyze the 3 million grocery orders from more than 200,000 Instacart users and predict which previously purchased item will be in user's next order</b>. Customer segmentation and affinity analysis are done to study customer purchase patterns and for better product marketing and cross-selling. <b>Achieved the best performance using the XGBoost model with an AUC score of 0.83, an accuracy of 0.74 and an F1-score of 0.36.</b></div>
<br>
<center><img src="images/marketbasket.png"></center>
<br>

---

### Song Clustering using K-Means

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/luthfiraditya/Song-Clustering-Using-K-Means){:target="_blank"}

<div style="text-align: justify">Using song data in the form of components in the song such as acousticness, energy, instrumentalness etc. Reduce data from 13 data variables into 8 components and can be used in K-Means modeling K-Means++. clustering songs into 8 n_clusters.</div>
<br>
<center><img src="images/songclustering.png"></center>
<br>

---

### Attrition/Turnover Prediction

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/luthfiraditya/Attrition-Turnover-Prediction){:target="_blank"}

<div style="text-align: justify"><b>In this project I conducted analysis and predictions related to turnover</b> on employee data in a company. Analysis is carried out to look for factors that cause turnover while predictions are used to predict which employees will make turnover. <b>This project also carried out several techniques such as normalization and sampling </b>(due to imbalanced data). After modeling using several classification models, especially the tree algorithm, it was found that <b>LightGBM produced the best performance with F1-score = 91% and ROC-AUC Score = 91%.This project uses kedro as a framework.</b></div>
<br>
<center><img src="images/attrition_designresearch.jpg"></center>
<br>

---
## Data Analyst

### Olist E-Commerce Business Performance

[![Read on Medium](https://img.shields.io/badge/GitHub-Read_on_Medium-white?logo=medium)](https://luthfirdty.medium.com/olist-e-commerce-business-performance-5ce0b3dc66fb){:target="_blank"}
[![Open Dashboard](https://img.shields.io/badge/Tableau-Open_Dashboard-orange?logo=tableau)](https://github.com/luthfiraditya/Olist-E-Commerce-Business-Performance){:target="_blank"}
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/luthfiraditya/Olist-E-Commerce-Business-Performance){:target="_blank"}


<div style="text-align: justify">In this project we have some objective to do about the bussiness performance inside Olist. I do analysis and visualization using tableau. <b>This analysis and visualization focuses on orders and transactions that occurred at olist during 2017 and 2018.</b></div>
<br>
<center><img src="images/sold_dash.png"/></center>
<br>

---
### Uber vs Green Cabs Trip in New York City - an Analysis

[![Open Dashboard](https://img.shields.io/badge/Tableau-Open_Dashboard-orange?logo=tableau)](https://public.tableau.com/app/profile/luthfi.raditya.meza/viz/UbervsGreenCabsTripinNewYorkCity/Dashboard1){:target="_blank"}
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/luthfiraditya/Uber-vs-Green-Cabs-Trip-in-New-York-City){:target="_blank"}



<div style="text-align: justify">Analyzing the performance of Green Cabs and Uber Taxi through visual analysis of passenger trips using Green Cabs and Uber Taxi from January to June in 2015 in the New York City area. Broadly speaking, there are two questions related to the research conducted:
<b>
<li>How do Green Cabs and Uber rides compare regionally in neighborhoods outside of New York City?</li>
<li>Do customer preferences change according to the time of day (night/day or weekend/weekday)?</li></b>
</div>
<br>
<center><img src="images/ubergreendash.png"/></center>
<br>

---

### Superstore Sales Dashboard

[![Open Dashboard](https://img.shields.io/badge/Tableau-Open_Dashboard-orange?logo=tableau)](https://public.tableau.com/app/profile/luthfi.raditya.meza/viz/SuperstoreSalesDashboard_16463843679940/SuperstoreDashboard){:target="_blank"}

<div style="text-align: justify">
Created a superstore dashboard showing the sales & profit by location, segment analysis, category analysis, shipping analysis in various years of a superstore. In this project also, I have made a interactive Tableau Sales Dashboard and find some insights from the data. </div>
<br>
<center><img src="images/SuperstoreDashboard.png"/></center>
<br>


---
<center>© 2023 Luthfi Raditya. Powered by Jekyll and the Minimal Theme.</center>
