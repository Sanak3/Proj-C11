"""
Módulo de Análise e Visualização de Dados - Projeto 1 (C11 - Inatel)
Dataset: AI Usage and Impact on Students & Professionals

Autores: Igor Araujo, Kevin Lopes do Vale, Felipe Fróes Pereira
"""

import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuração visual padrão para gráficos elegantes e consistentes
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 100


def carregar_e_limpar_dados(
    caminho_csv: str = "data/AI_Usage_and_Impact_on_Students_and_Professionals.csv",
) -> pd.DataFrame:
    """Carrega o dataset bruto, remove duplicatas, padroniza categorias,

    converte tipos de dados e realiza engenharia de atributos.
    """
    # Busca inteligente do arquivo para funcionar tanto da raiz quanto da pasta notebooks/
    caminhos_tentativa = [
        caminho_csv,
        os.path.join("..", caminho_csv),
        caminho_csv.lstrip("../"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "AI_Usage_and_Impact_on_Students_and_Professionals.csv"),
    ]
    caminho_final = None
    for c in caminhos_tentativa:
        if os.path.exists(c):
            caminho_final = c
            break

    if not caminho_final:
        raise FileNotFoundError(
            f"Arquivo não encontrado nos caminhos testados. "
            "Certifique-se de que o arquivo CSV está na pasta 'data/'."
        )

    df = pd.read_csv(caminho_final)

    # 1. Remoção de duplicatas exatas
    df = df.drop_duplicates().copy()

    # 2. Conversão e higienização de colunas numéricas
    def extrair_float(serie: pd.Series) -> pd.Series:
        texto = (
            serie.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        extraido = texto.str.extract(r"(\d+(?:\.\d+)?)")[0]
        return pd.to_numeric(extraido, errors="coerce")

    colunas_numericas = [
        "Age",
        "Monthly_Income",
        "AI_Usage_Hours_Per_Day",
        "Monthly_AI_Cost",
        "Productivity_Score",
        "Accuracy_Rating",
        "Satisfaction_Score",
        "Time_Saved_Hours_Per_Week",
        "Work_or_Study_Hours_Per_Day",
    ]
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = extrair_float(df[col])

    # 3. Padronização da ferramenta de IA (AI_Tool)
    mapa_ferramentas = {
        "chatgpt": "ChatGPT",
        "chat gpt": "ChatGPT",
        "google gemini": "Google Gemini",
        "gemini": "Google Gemini",
        "microsoft copilot": "Microsoft Copilot",
        "copilot": "Microsoft Copilot",
        "claude": "Claude",
        "perplexity ai": "Perplexity AI",
        "meta ai": "Meta AI",
        "deepseek": "DeepSeek",
        "grammarly": "Grammarly",
    }
    df["AI_Tool"] = (
        df["AI_Tool"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapa_ferramentas)
        .fillna(df["AI_Tool"])
    )
    df["AI_Tool"] = df["AI_Tool"].replace({"nan": np.nan, "None": np.nan})

    # 4. Padronização de Gênero (Gender)
    mapa_genero = {
        "male": "Male",
        "m": "Male",
        "female": "Female",
        "f": "Female",
        "other": "Other",
        "non-binary": "Other",
    }
    df["Gender"] = (
        df["Gender"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapa_genero)
        .fillna(df["Gender"])
    )
    df["Gender"] = df["Gender"].replace({"nan": np.nan, "None": np.nan})

    # 5. Padronização de País (Country)
    df["Country"] = df["Country"].astype(str).str.strip().str.title()
    df["Country"] = df["Country"].replace(
        {
            "United States Of America": "United States",
            "U.K.": "United Kingdom",
            "Nan": np.nan,
        }
    )

    # 6. Padronização de Nível de Escolaridade (Education_Level)
    mapa_escolaridade = {
        "masters": "Master's",
        "master's": "Master's",
        "master's degree": "Master's",
        "ma/msc": "Master's",
        "bachelors": "Bachelor's",
        "bachelor's": "Bachelor's",
        "bachelor's degree": "Bachelor's",
        "ba/bsc": "Bachelor's",
        "undergraduate": "Undergraduate",
        "undergrad": "Undergraduate",
        "high school": "High School",
        "hs": "High School",
        "phd": "PhD",
        "ph.d.": "PhD",
        "diploma": "Diploma",
    }
    df["Education_Level"] = (
        df["Education_Level"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapa_escolaridade)
        .fillna(df["Education_Level"])
    )
    df["Education_Level"] = df["Education_Level"].replace(
        {"nan": np.nan, "None": np.nan}
    )

    # 7. Engenharia de Atributos (Feature Engineering)
    # Eficiência: Horas economizadas por dia dividido por horas gastas com IA
    df["Time_Saved_Per_Usage_Hour"] = np.where(
        (df["AI_Usage_Hours_Per_Day"] > 0)
        & (df["Time_Saved_Hours_Per_Week"].notnull()),
        (df["Time_Saved_Hours_Per_Week"] / 7.0) / df["AI_Usage_Hours_Per_Day"],
        np.nan,
    )

    # Segmentação de Custo: Usuário Gratuito ($0) vs Usuário Pago (> $0)
    df["Paid_User"] = np.where(df["Monthly_AI_Cost"] > 0, "Pago (> $0)", "Gratuito ($0)")
    df.loc[df["Monthly_AI_Cost"].isna(), "Paid_User"] = np.nan

    return df


# ==============================================================================
# PERGUNTAS 1 A 5 (IMPLEMENTADAS)
# ==============================================================================


def analisar_pergunta_1(
    df: pd.DataFrame, salvar_grafico: Optional[str] = None
) -> Tuple[pd.DataFrame, plt.Figure]:
    """Pergunta 1: Estudantes vs.

    Profissionais Comparação de horas diárias gastas com IA e pontuação de
    produtividade. Gráfico: Boxplot comparativo lado a lado (2 subplots).
    """
    resumo = (
        df.groupby("User_Type")[
            ["AI_Usage_Hours_Per_Day", "Productivity_Score"]
        ]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1.1: Horas de uso por dia
    sns.boxplot(
        data=df,
        x="User_Type",
        y="AI_Usage_Hours_Per_Day",
        hue="User_Type",
        legend=False,
        palette=["#3470a3", "#e27c38"],
        ax=axes[0],
    )
    axes[0].set_title(
        "Horas Diárias de Uso de IA por Perfil", fontsize=13, fontweight="bold"
    )
    axes[0].set_xlabel("Perfil do Usuário", fontsize=11)
    axes[0].set_ylabel("Uso Diário (horas)", fontsize=11)

    # Gráfico 1.2: Produtividade reportada
    sns.boxplot(
        data=df,
        x="User_Type",
        y="Productivity_Score",
        hue="User_Type",
        legend=False,
        palette=["#3470a3", "#e27c38"],
        ax=axes[1],
    )
    axes[1].set_title(
        "Pontuação de Produtividade Percebida por Perfil",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xlabel("Perfil do Usuário", fontsize=11)
    axes[1].set_ylabel("Productivity Score (0 a 10)", fontsize=11)

    plt.tight_layout()
    if salvar_grafico:
        fig.savefig(salvar_grafico, bbox_inches="tight")
    return resumo, fig


def analisar_pergunta_2(
    df: pd.DataFrame, salvar_grafico: Optional[str] = None
) -> Tuple[pd.DataFrame, plt.Figure]:
    """Pergunta 2: Preferência de Ferramentas Identificação das IAs mais

    populares e sua distribuição entre perfis. Gráfico: Barras horizontais
    agrupadas por perfil de usuário.
    """
    tabela_cruzada = pd.crosstab(
        df["AI_Tool"], df["User_Type"], margins=True
    ).sort_values("All", ascending=False)

    df_plot = (
        df.dropna(subset=["AI_Tool", "User_Type"])
        .groupby(["AI_Tool", "User_Type"])
        .size()
        .reset_index(name="Quantidade")
    )

    # Ordenar pelas ferramentas com maior volume total
    ordem_ferramentas = (
        df["AI_Tool"].value_counts().sort_values(ascending=True).index
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_plot,
        y="AI_Tool",
        x="Quantidade",
        hue="User_Type",
        order=ordem_ferramentas,
        palette=["#3470a3", "#e27c38"],
        ax=ax,
    )

    ax.set_title(
        "Preferência de Ferramentas de IA: Estudantes vs. Profissionais",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Número de Usuários", fontsize=11)
    ax.set_ylabel("Ferramenta de IA", fontsize=11)
    ax.legend(title="Perfil de Usuário")

    plt.tight_layout()
    if salvar_grafico:
        fig.savefig(salvar_grafico, bbox_inches="tight")
    return tabela_cruzada, fig


def analisar_pergunta_3(
    df: pd.DataFrame, salvar_grafico: Optional[str] = None
) -> Tuple[dict, plt.Figure]:
    """Pergunta 3: Tempo de Uso vs.

    Economia Real Correlação entre horas diárias gastas e economia semanal,
    avaliando retornos decrescentes. Gráfico: Gráfico de dispersão (Scatter) com
    linha de regressão e curva de saturação.
    """
    dados_validos = df[
        ["AI_Usage_Hours_Per_Day", "Time_Saved_Hours_Per_Week"]
    ].dropna()
    x = dados_validos["AI_Usage_Hours_Per_Day"]
    y = dados_validos["Time_Saved_Hours_Per_Week"]

    corr_pearson = round(float(x.corr(y, method="pearson")), 3)
    corr_spearman = round(float(x.rank().corr(y.rank())), 3)

    # Ajuste polinomial de 2º grau para evidenciar eventual curvatura de saturação
    poly_coefs = np.polyfit(x, y, deg=2)
    poly_fn = np.poly1d(poly_coefs)

    x_linha = np.linspace(x.min(), x.max(), 100)
    y_linha = poly_fn(x_linha)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(
        x=x,
        y=y,
        scatter_kws={"alpha": 0.6, "color": "#1f77b4"},
        line_kws={
            "color": "#d62728",
            "linewidth": 2,
            "label": f"Tendência Linear (Pearson r = {corr_pearson})",
        },
        ax=ax,
    )
    ax.plot(
        x_linha,
        y_linha,
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        label="Ajuste Polinomial (Grau 2)",
    )

    ax.set_title(
        "Relação entre Horas Diárias de IA e Horas Semanais Economizadas",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Horas Diárias Usando IA", fontsize=11)
    ax.set_ylabel("Horas Semanais Economizadas", fontsize=11)
    ax.legend(loc="upper left")

    metricas = {
        "Pearson_r": corr_pearson,
        "Spearman_r": corr_spearman,
        "Coeficientes_Polinomio": poly_coefs.tolist(),
        "Total_Amostras": len(dados_validos),
    }

    plt.tight_layout()
    if salvar_grafico:
        fig.savefig(salvar_grafico, bbox_inches="tight")
    return metricas, fig


def analisar_pergunta_4(
    df: pd.DataFrame, salvar_grafico: Optional[str] = None
) -> Tuple[pd.DataFrame, plt.Figure]:
    """Pergunta 4: Finalidade de Uso e Satisfação Ranking dos casos de uso

    (Coding, Writing, Research, etc.) por notas médias de satisfação e acurácia.
    Gráfico: Mapa de Calor (Heatmap) com matriz de médias.
    """
    resumo_proposito = (
        df.groupby("AI_Purpose")[["Satisfaction_Score", "Accuracy_Rating"]]
        .agg(["mean", "count"])
        .round(2)
    )

    # Matriz para heatmap ordenada pela satisfação
    matriz_medias = (
        df.groupby("AI_Purpose")[["Satisfaction_Score", "Accuracy_Rating"]]
        .mean()
        .sort_values("Satisfaction_Score", ascending=False)
        .round(2)
    )
    matriz_medias.columns = ["Satisfação Média", "Acurácia Média"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matriz_medias,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Nota Média"},
        ax=ax,
    )

    ax.set_title(
        "Satisfação e Acurácia Média por Caso de Uso de IA",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Métrica Avaliada", fontsize=11)
    ax.set_ylabel("Propósito de Uso (AI Purpose)", fontsize=11)

    plt.tight_layout()
    if salvar_grafico:
        fig.savefig(salvar_grafico, bbox_inches="tight")
    return resumo_proposito, fig


def analisar_pergunta_5(
    df: pd.DataFrame, salvar_grafico: Optional[str] = None
) -> Tuple[pd.DataFrame, plt.Figure]:
    """Pergunta 5: Investimento Financeiro em IA (Planos Pagos vs.

    Gratuitos) Avaliação se usuários que investem financeiramente em planos
    pagos (Monthly_AI_Cost > 0) reportam maior produtividade e maior economia de
    tempo. Gráfico: Boxplot comparativo entre Usuários Pagos vs Gratuitos.
    """
    resumo_custo = (
        df.groupby("Paid_User")[
            [
                "Productivity_Score",
                "Time_Saved_Hours_Per_Week",
                "Monthly_AI_Cost",
            ]
        ]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )

    df_validos = df.dropna(
        subset=["Paid_User", "Productivity_Score", "Time_Saved_Hours_Per_Week"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 5.1: Produtividade por categoria de custo
    sns.boxplot(
        data=df_validos,
        x="Paid_User",
        y="Productivity_Score",
        hue="Paid_User",
        legend=False,
        palette=["#2b9348", "#55a630"],
        ax=axes[0],
    )
    axes[0].set_title(
        "Produtividade: Usuários Gratuitos vs. Pagos",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].set_xlabel("Modalidade de Uso", fontsize=11)
    axes[0].set_ylabel("Productivity Score (0 a 10)", fontsize=11)

    # Gráfico 5.2: Economia de tempo por categoria de custo
    sns.boxplot(
        data=df_validos,
        x="Paid_User",
        y="Time_Saved_Hours_Per_Week",
        hue="Paid_User",
        legend=False,
        palette=["#2b9348", "#55a630"],
        ax=axes[1],
    )
    axes[1].set_title(
        "Tempo Economizado Semanal: Gratuitos vs. Pagos",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xlabel("Modalidade de Uso", fontsize=11)
    axes[1].set_ylabel("Horas Semanais Economizadas", fontsize=11)

    plt.tight_layout()
    if salvar_grafico:
        fig.savefig(salvar_grafico, bbox_inches="tight")
    return resumo_custo, fig


# ==============================================================================
# PERGUNTAS 6 A 10 (TEMPLATES E INSTRUÇÕES PARA O AMIGO CODAR)
# ==============================================================================


def analisar_pergunta_6(df: pd.DataFrame):
    """TODO (Amigo): Pergunta 6 - Acurácia Percebida por Ferramenta

    Pergunta: Existe discrepância estatística na acurácia reportada
    (Accuracy_Rating)
              entre as diferentes ferramentas de IA (ChatGPT, Gemini, Claude, Copilot, etc.)?

    Variáveis:
        - AI_Tool (categórica)
        - Accuracy_Rating (numérica, notas de 1 a 5)

    Sugestão de Gráfico:
        - Boxplot ou Violin Plot (sns.boxplot ou sns.violinplot) com ordenação pela mediana.
    """
    pass


def analisar_pergunta_7(df: pd.DataFrame):
    """TODO (Amigo): Pergunta 7 - Renda Mensal e Áreas de Atuação

    Pergunta: Profissionais de áreas que dependem fortemente de IA (Tecnologia,
    Dados)
              apresentam rendas mensais (Monthly_Income) superiores aos que usam para suporte?

    Variáveis:
        - Monthly_Income (numérica)
        - Profession ou AI_Purpose (categórica)
        - Filtrar preferencialmente apenas 'User_Type == "Professional"'

    Sugestão de Gráfico:
        - Gráfico de barras horizontais com renda média/mediana (sns.barplot).
    """
    pass


def analisar_pergunta_8(df: pd.DataFrame):
    """TODO (Amigo): Pergunta 8 - Nível Educacional e Adoção para Pesquisa

    Pergunta: O uso de IA para Research e Studying é mais prevalente e avaliado de forma
              diferente entre estudantes de graduação versus pós-graduação/mestrado/PhD?

    Variáveis:
        - Education_Level (categórica: 'Undergraduate', 'Bachelor\'s', 'Master\'s', 'PhD')
        - AI_Purpose (filtrar 'Research' e 'Studying')
        - Satisfaction_Score / Accuracy_Rating

    Sugestão de Gráfico:
        - Gráfico de barras empilhadas ou agrupadas mostrando % de uso para Research/Studying.
    """
    pass


def analisar_pergunta_9(df: pd.DataFrame):
    """TODO (Amigo): Pergunta 9 - Impacto no Atendimento e Rotinas Corporativas

    Pergunta: No setor de Customer Support e Office Work, qual é a média de tempo semanal
              economizado e qual o impacto percebido na produtividade?

    Variáveis:
        - AI_Purpose.isin(['Customer Support', 'Office Work'])
        - Time_Saved_Hours_Per_Week
        - Productivity_Score

    Sugestão de Gráfico:
        - Gráfico de barras com erro padrão ou Pointplot comparando os dois setores com a média global.
    """
    pass


def analisar_pergunta_10(df: pd.DataFrame):
    """TODO (Amigo): Pergunta 10 - Faixa Etária e Padrão de Adoção

    Pergunta: Como a idade (Age) influencia a escolha da ferramenta de IA e a percepção geral
              de satisfação com as soluções adotadas?

    Variáveis:
        - Age (numérica: pode criar faixas como '<20', '20-29', '30-39', '40+')
        - AI_Tool
        - Satisfaction_Score

    Sugestão de Gráfico:
        - Dispersão com jitter ou Boxplot de Satisfação por faixa etária / Ferramenta mais usada por idade.
    """
    pass
