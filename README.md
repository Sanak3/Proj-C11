# Projeto 1 - C11: Ciência de Dados
**Instituto Nacional de Telecomunicações (Inatel)**  
**Docente:** Prof. Dr. Renzo Mesquita Paranaíba  

---

## Integrantes do Grupo
| Nome                   | Matrícula | Curso |
|:-----------------------|:----------|:------|
| *Igor Araujo*          | *479*     | GES   |
| *Kevin Lopes do vale*  | *1955*    | GEC   |
| *Felipe Fróes Pereira* | *566*     | GES   |

---

## Sobre o Projeto

Este projeto consiste na aplicação prática do pipeline de **Ciência de Dados** para analisar e extrair insights valiosos a partir de um dataset real selecionado do Kaggle. O trabalho aborda etapas essenciais que vão desde o pré-processamento e higienização dos dados até a análise exploratória (EDA) e visualização gráfica, utilizando a biblioteca padrão do ecossistema de dados em Python.

---

## Dataset Selecionado

- **Nome:** [AI Usage and Impact on Students & Professionals](https://www.kaggle.com/datasets/eishatuzzuhra/ai-usage-and-impact-on-students-and-professionals)
- **Autor(a):** Eisha Tuz Zuhra (Kaggle)
- **Formato:** Arquivo CSV (`AI_Usage_and_Impact_on_Students_and_Professionals.csv`)
- **Volume:** 300 registros brutos (284 registros analisados após deduplicação e filtro de sanidade de outliers), 19 variáveis (categóricas e numéricas)
- **Licença:** CC0: Public Domain

### Contextualização do Problema
Com a explosão de ferramentas de Inteligência Artificial Generativa (como ChatGPT, Claude, Google Gemini, Perplexity, Copilot), estudantes e profissionais de diversas indústrias transformaram suas rotinas de estudo e trabalho. O objetivo deste estudo é analisar **como a IA está sendo adotada**, quais ferramentas e propósitos predominam em cada perfil, e **qual é o impacto real** dessa tecnologia sobre produtividade, economia de tempo e precisão percebida.

### Principais Variáveis do Dataset

| Categoria | Variáveis | Descrição |
| :--- | :--- | :--- |
| **Demográficas** | `User_ID`, `Age`, `Gender`, `Country`, `Education_Level`, `Profession`, `Monthly_Income` | Perfil sociodemográfico, financeiro e acadêmico/profissional do entrevistado. |
| **Perfil de Usuário** | `User_Type` | Classificação do usuário (`Student` ou `Professional`). |
| **Comportamento com IA** | `AI_Tool`, `AI_Purpose`, `AI_Usage_Hours_Per_Day`, `Tasks_Performed`, `Monthly_AI_Cost` | Ferramenta principal utilizada, objetivo de uso, horas diárias dedicadas, tarefas executadas e investimento financeiro mensal em IA. |
| **Rotina e Recomendação** | `Work_or_Study_Hours_Per_Day`, `Would_Recommend` | Carga horária diária de trabalho/estudo e recomendação da ferramenta para terceiros. |
| **Impacto e Métricas** | `Productivity_Score`, `Accuracy_Rating`, `Satisfaction_Score`, `Time_Saved_Hours_Per_Week` | Avaliações numéricas de ganhos de produtividade, acurácia percebida, satisfação geral e horas poupadas semanalmente. |

---

## As 10 Perguntas de Pesquisa & Negócio

Para direcionar a análise exploratória e gerar insights estratégicos, o projeto investiga as seguintes 10 perguntas-chave:

1. **Estudantes vs. Profissionais:** Qual dos dois perfis despende mais horas diárias utilizando ferramentas de IA (`AI_Usage_Hours_Per_Day`) e quem reporta maior índice de produtividade (`Productivity_Score`)?
2. **Preferência de Ferramentas:** Quais são as ferramentas de IA mais populares no dataset e como essa distribuição se divide entre estudantes e profissionais?
3. **Tempo de Uso vs. Economia Real:** Existe uma correlação positiva direta entre as horas diárias gastas com IA (`AI_Usage_Hours_Per_Day`) e o tempo semanal economizado (`Time_Saved_Hours_Per_Week`), ou há um ponto de estagnação/retornos decrescentes?
4. **Finalidade de Uso e Satisfação:** Quais casos de uso (`AI_Purpose` – Ex: Coding, Writing, Research, Data Analysis) apresentam os maiores índices de satisfação (`Satisfaction_Score`) e acurácia percebida (`Accuracy_Rating`)?
5. **Investimento em IA (Planos Pagos vs. Gratuitos):** Usuários que investem financeiramente em ferramentas de IA (`Monthly_AI_Cost`) conseguem obter notas significativamente maiores de produtividade e economia de tempo em comparação aos que utilizam apenas versões gratuitas?
6. **Acurácia Percebida por Ferramenta:** Existe discrepância estatística na acurácia reportada (`Accuracy_Rating`) entre as diferentes ferramentas de IA (ChatGPT, Gemini, Claude, Copilot, DeepSeek, etc.)?
7. **Renda Mensal e Áreas de Atuação:** Profissionais de áreas que dependem fortemente de IA (como Tecnologia, Dados e Desenvolvimento) apresentam rendas mensais (`Monthly_Income`) superiores àqueles que usam para suporte ou tarefas administrativas?
8. **Nível Educacional e Adoção de IA para Pesquisa:** O uso de IA para *Research* e *Studying* é mais prevalente e avaliado de forma diferente entre estudantes de graduação versus pós-graduação/mestrado/PhD?
9. **Impacto no Atendimento e Rotinas Corporativas:** No setor de *Customer Support* e *Office Work*, qual é a média de tempo semanal economizado e qual o impacto percebido na produtividade?
10. **Faixa Etária e Padrão de Adoção:** Como a idade (`Age`) influencia a escolha da ferramenta de IA e a percepção geral de satisfação com as soluções adotadas?

---

## Tecnologias e Ferramentas

O projeto é desenvolvido integralmente em **Python**, utilizando o conjunto de bibliotecas estudadas na disciplina:

- **[Python 3.10+](https://www.python.org/):** Linguagem base do projeto.
- **[Pandas](https://pandas.pydata.org/):** Manipulação, tratamento de valores nulos e estruturação dos dados tabulares.
- **[NumPy](https://numpy.org/):** Operações numéricas, cálculos estatísticos e transformações de matrizes/vetores.
- **[Matplotlib](https://matplotlib.org/):** Criação e customização básica dos gráficos analíticos.
- **[Seaborn](https://seaborn.pydata.org/):** Visualizações estatísticas refinadas (gráficos de distribuição, dispersão, boxplots e heatmaps).
- **[Jupyter Notebook](https://jupyter.org/):** Ambiente interativo para documentação, código e exibição dos resultados.

---

## Estrutura do Repositório

```text
├── data/
│   └── AI_Usage_and_Impact_on_Students_and_Professionals.csv  # Dataset original do Kaggle
├── notebooks/
│   └── analise_projeto_c11.ipynb                             # Notebook com limpeza, EDA e gráficos
├── src/
│   └── analises.py                                           # Scripts auxiliares e funções de análise
├── slides/
│   └── Apresentacao_Resultados_Pitch.pdf                     # Slides da apresentação de 5 minutos
├── .gitignore
├── README.md
└── requirements.txt                                          # Dependências do projeto
```

---

## Como Executar o Projeto

### 1. Clonar o repositório
```bash
git clone https://github.com/Sanak3/Proj-C11
cd Proj-C11
```

### 2. Criar e ativar um ambiente virtual (recomendado)
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4. Executar as análises
Você pode abrir o notebook interativo no Jupyter:
```bash
jupyter notebook notebooks/analise_projeto_c11.ipynb
```
Ou executar o script Python diretamente via terminal:
```bash
python main.py
```

---

## Etapas da Análise de Dados

1. **Entendimento do Problema e Exploração Inicial:** Inspeção das 19 colunas, tipos de dados, valores ausentes e inconsistências de formatação.
2. **Limpeza e Pré-processamento:** Padronização de strings (ex: nomes de ferramentas e países), tratamento de valores ausentes e conversão de tipos numéricos.
3. **Engenharia de Recursos (Feature Engineering):** Criação de métricas derivadas (ex: relação de tempo poupado por hora de uso).
4. **Análise Exploratória e Resposta às 10 Perguntas:** Elaboração de tabelas descritivas e cruzamento de variáveis categóricas e numéricas.
5. **Visualização de Dados:** Gráficos com paletas adequadas, títulos informativos e conclusões voltadas à tomada de decisão.
6. **Conclusões e Insights:** Síntese dos achados e geração de valor a partir dos dados.

---

## Licença e Referências

- Dataset original disponibilizado por **Eisha Tuz Zuhra** sob a licença [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) no Kaggle.
- Projeto desenvolvido para fins estritamente acadêmicos no curso de Engenharia do **Inatel**.