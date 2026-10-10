# Projeto 1 - C11: Ciência de Dados
**Instituto Nacional de Telecomunicações (Inatel)**  
**Docente:** Prof. Dr. Renzo Mesquita Paranaíba  

---

## 📌 Sobre o Projeto

Este projeto consiste na aplicação prática do pipeline de **Ciência de Dados** para analisar e extrair insights valiosos a partir de um dataset real selecionado do Kaggle. O trabalho aborda etapas essenciais que vão desde o pré-processamento e higienização dos dados até a análise exploratória (EDA) e visualização gráfica, utilizando a biblioteca padrão do ecossistema de dados em Python.

---

## 📊 Dataset Selecionado

- **Nome:** [AI Usage and Impact on Students & Professionals](https://www.kaggle.com/datasets/eishatuzzuhra/ai-usage-and-impact-on-students-and-professionals)
- **Autor(a):** Eisha Tuz Zuhra (Kaggle)
- **Formato:** Arquivo CSV (`AI_Usage_and_Impact_on_Students_and_Professionals.csv`)
- **Volume:** Mais de 1.000 registros, 19 variáveis (categóricas e numéricas)
- **Licença:** CC0: Public Domain

### Contextualização do Problema
Com a explosão de ferramentas de Inteligência Artificial Generativa (como ChatGPT, Claude, Google Gemini, Perplexity, Copilot), estudantes e profissionais de diversas indústrias transformaram suas rotinas de estudo e trabalho. O objetivo deste estudo é analisar **como a IA está sendo adotada**, quais ferramentas e propósitos predominam em cada perfil, e **qual é o impacto real** dessa tecnologia sobre produtividade, economia de tempo e precisão percebida.

### Principais Variáveis do Dataset

| Categoria | Variáveis | Descrição |
| :--- | :--- | :--- |
| **Demográficas** | `User_ID`, `Age`, `Gender`, `Country`, `Education_Level`, `Profession`, `Monthly_Income` | Perfil sociodemográfico, financeiro e acadêmico/profissional do entrevistado. |
| **Perfil de Usuário** | `User_Type` | Classificação do usuário (`Student` ou `Professional`). |
| **Comportamento com IA** | `AI_Tool`, `AI_Purpose`, `AI_Usage_Hours_Per_Day`, `Tasks_Performed` | Ferramenta principal utilizada, objetivo de uso, horas diárias dedicadas e tarefas executadas. |
| **Impacto e Métricas** | `Productivity_Score`, `Accuracy_Rating`, `Satisfaction_Score`, `Time_Saved_Hours_Per_Week`, `AI_Experience_Years` | Avaliações numéricas de ganhos de produtividade, acurácia, satisfação, horas poupadas semanalmente e anos de experiência com IA. |