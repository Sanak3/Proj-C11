"""
Módulo de Análise e Visualização de Dados - Projeto 1 (C11 - Inatel)
Dataset: AI Usage and Impact on Students & Professionals

Autores: Igor Araujo, Kevin Lopes do Vale, Felipe Fróes Pereira
"""

import os
from typing import Any, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr


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

    # 8. Filtro de Sanidade e Limpeza de Outliers Extremos
    # Remove registros espúrios com erros de digitação (ex: 23h diárias de IA, 97 anos em estudantes, renda mensal de $250k)
    mascara_sanidade = (
        (df["AI_Usage_Hours_Per_Day"].isna() | (df["AI_Usage_Hours_Per_Day"] <= 16.0))
        & (df["Monthly_Income"].isna() | (df["Monthly_Income"] <= 50000.0))
        & (df["Age"].isna() | (df["Age"] <= 80.0))
    )
    df = df[mascara_sanidade].copy()

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
# PERGUNTAS 6 A 10 (IMPLEMENTADAS COM TESTES DE HIPÓTESES ESTATÍSTICOS)
# ==============================================================================


def analisar_pergunta_6(
    df: pd.DataFrame, salvar_grafico: Optional[str] = None
) -> Tuple[pd.DataFrame, plt.Figure]:
    """Pergunta 6: Acurácia Percebida por Ferramenta de IA.

    Avalia se existe discrepância estatisticamente significativa na acurácia
    reportada (Accuracy_Rating) entre as diferentes ferramentas de IA através
    do teste não-paramétrico de Kruskal-Wallis.
    Gráfico: Boxplot de Acurácia Percebida por Ferramenta (ordenado pela mediana).
    """
    dados = df.dropna(subset=["AI_Tool", "Accuracy_Rating"]).copy()

    # Teste estatístico não-paramétrico de Kruskal-Wallis
    grupos = [grupo["Accuracy_Rating"].values for _, grupo in dados.groupby("AI_Tool")]
    res_kruskal = kruskal(*grupos)

    resumo = (
        dados.groupby("AI_Tool")["Accuracy_Rating"]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )
    resumo.columns = ["Contagem", "Média", "Mediana", "Desvio_Padrão"]
    resumo["Kruskal_H"] = round(float(res_kruskal.statistic), 3)
    resumo["Valor_p"] = round(float(res_kruskal.pvalue), 4)

    ordem_mediana = (
        dados.groupby("AI_Tool")["Accuracy_Rating"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=dados,
        x="AI_Tool",
        y="Accuracy_Rating",
        order=ordem_mediana,
        palette="Blues_r",
        hue="AI_Tool",
        legend=False,
        ax=ax,
    )

    sig_texto = (
        "Diferença estatisticamente significativa (p < 0.05)"
        if res_kruskal.pvalue < 0.05
        else "Sem diferença estatisticamente significativa (p >= 0.05)"
    )

    ax.set_title(
        f"Acurácia Percebida por Ferramenta de IA\n"
        f"Kruskal-Wallis: H = {res_kruskal.statistic:.2f}, p-valor = {res_kruskal.pvalue:.3f} ({sig_texto})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Ferramenta de IA", fontsize=11)
    ax.set_ylabel("Acurácia Percebida (1 a 5)", fontsize=11)
    plt.xticks(rotation=25)

    plt.tight_layout()
    if salvar_grafico:
        fig.savefig(salvar_grafico, bbox_inches="tight")
    return resumo, fig


def analisar_pergunta_7(
    df: pd.DataFrame, salvar_grafico: Optional[str] = None
) -> Tuple[pd.DataFrame, plt.Figure]:
    """Pergunta 7: Renda Mensal e Áreas de Atuação.

    Compara rendas mensais (Monthly_Income) entre profissionais de Tecnologia/Dados/Desenvolvimento
    versus Suporte/Administrativo via teste de Mann-Whitney U.
    Gráfico: Boxplot comparativo com distribuição das rendas.
    """
    df_prof = df[df["User_Type"] == "Professional"].copy()
    profissao = df_prof["Profession"].fillna("")

    suporte_mask = profissao.str.contains(
        r"Customer|Client|Kundenservice|Kundenbetreuer|Support|Service|Success|Experience|Administrative|Data Entry|Assistente|Atendimento",
        case=False,
        regex=True,
    )
    tecnologia_mask = profissao.str.contains(
        r"Data|Daten|Software|Developer|Développeur|Softwareentwickler|Engineer|Ingenieur|Technology|Computer|Machine Learning|Desenvolvedor|Desenvolvedora|Engenheiro|Analista|Analyste|Intelligence|AI",
        case=False,
        regex=True,
    )

    df_prof["Area"] = "Outras"
    df_prof.loc[tecnologia_mask, "Area"] = "Tecnologia/Dados"
    df_prof.loc[suporte_mask, "Area"] = "Suporte/Admin"

    dados = df_prof[df_prof["Area"].isin(["Tecnologia/Dados", "Suporte/Admin"])].dropna(
        subset=["Monthly_Income"]
    ).copy()

    renda_tec = dados[dados["Area"] == "Tecnologia/Dados"]["Monthly_Income"]
    renda_sup = dados[dados["Area"] == "Suporte/Admin"]["Monthly_Income"]

    res_mwu = mannwhitneyu(renda_tec, renda_sup)

    resumo = (
        dados.groupby("Area")["Monthly_Income"]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )
    resumo.columns = ["Contagem", "Média", "Mediana", "Desvio_Padrão"]
    resumo["MannWhitney_U"] = round(float(res_mwu.statistic), 2)
    resumo["Valor_p"] = round(float(res_mwu.pvalue), 4)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=dados,
        x="Area",
        y="Monthly_Income",
        palette=["#3470a3", "#e27c38"],
        hue="Area",
        legend=False,
        ax=ax,
    )

    sig_texto = (
        "Diferença significativa (p < 0.05)"
        if res_mwu.pvalue < 0.05
        else "Sem diferença significativa (p >= 0.05)"
    )

    ax.set_title(
        f"Renda Mensal: Tecnologia/Dados vs. Suporte/Administrativo\n"
        f"Mann-Whitney U: p-valor = {res_mwu.pvalue:.3f} ({sig_texto})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Área Profissional", fontsize=11)
    ax.set_ylabel("Renda Mensal ($)", fontsize=11)

    plt.tight_layout()
    if salvar_grafico:
        fig.savefig(salvar_grafico, bbox_inches="tight")
    return resumo, fig

def analisar_pergunta_8(
    df: pd.DataFrame, salvar_grafico: Optional[str] = None
) -> Tuple[Dict[str, Any], plt.Figure]:
    """Pergunta 8: Nível Educacional e Adoção para Pesquisa.

    Analisa a prevalência de uso de IA para Research/Studying (teste Qui-Quadrado)
    e a avaliação de Acurácia e Satisfação (testes Mann-Whitney U) entre
    estudantes de Graduação e Pós-graduação.
    Gráfico: Painel com 3 subplots (Prevalência %, Acurácia e Satisfação).
    """
    mapa_nivel = {
        "Bachelor's": "Graduação",
        "Undergraduate": "Graduação",
        "Master's": "Pós-graduação",
        "PhD": "Pós-graduação",
    }
    df_estudante = df[df["User_Type"] == "Student"].copy()
    df_estudante["Nivel"] = df_estudante["Education_Level"].map(mapa_nivel)

    dados = df_estudante[
        df_estudante["Nivel"].isin(["Graduação", "Pós-graduação"])
    ].dropna(subset=["AI_Purpose"]).copy()

    # 8A: Tabela de contingência e Qui-Quadrado para Research/Studying
    dados["Usa_Pesquisa_Estudo"] = dados["AI_Purpose"].isin(["Research", "Studying"])
    tabela_qui2 = pd.crosstab(dados["Nivel"], dados["Usa_Pesquisa_Estudo"])
    res_qui2 = chi2_contingency(tabela_qui2)

    # Proporções por finalidade para o gráfico 8A
    tabela_prop = (
        pd.crosstab(dados["Nivel"], dados["AI_Purpose"], normalize="index") * 100
    ).round(2)

    # 8B e 8C: Acurácia e Satisfação filtrando apenas quem usa para Research/Studying
    dados_pesquisa = dados[dados["Usa_Pesquisa_Estudo"]].copy()

    acc_grad = dados_pesquisa[dados_pesquisa["Nivel"] == "Graduação"]["Accuracy_Rating"].dropna()
    acc_pos = dados_pesquisa[dados_pesquisa["Nivel"] == "Pós-graduação"]["Accuracy_Rating"].dropna()
    res_mwu_acc = mannwhitneyu(acc_grad, acc_pos)

    sat_grad = dados_pesquisa[dados_pesquisa["Nivel"] == "Graduação"]["Satisfaction_Score"].dropna()
    sat_pos = dados_pesquisa[dados_pesquisa["Nivel"] == "Pós-graduação"]["Satisfaction_Score"].dropna()
    res_mwu_sat = mannwhitneyu(sat_grad, sat_pos)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Gráfico 8A: Barras de proporção (%)
    colunas_foco = [c for c in ["Research", "Studying"] if c in tabela_prop.columns]
    tabela_prop[colunas_foco].plot(
        kind="bar",
        ax=axes[0],
        color=["#2b5c8f", "#d95f02"],
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_title(
        f"Adoção para Pesquisa e Estudo\n(Qui-Quadrado p = {res_qui2.pvalue:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Nível Educacional", fontsize=10)
    axes[0].set_ylabel("Percentual de Estudantes (%)", fontsize=10)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Finalidade")

    # Gráfico 8B: Boxplot de Acurácia
    sns.boxplot(
        data=dados_pesquisa,
        x="Nivel",
        y="Accuracy_Rating",
        palette=["#3470a3", "#e27c38"],
        hue="Nivel",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title(
        f"Acurácia Percebida (Pesquisa/Estudo)\n(Mann-Whitney p = {res_mwu_acc.pvalue:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Nível Educacional", fontsize=10)
    axes[1].set_ylabel("Accuracy Rating (1 a 5)", fontsize=10)

    # Gráfico 8C: Boxplot de Satisfação
    sns.boxplot(
        data=dados_pesquisa,
        x="Nivel",
        y="Satisfaction_Score",
        palette=["#3470a3", "#e27c38"],
        hue="Nivel",
        legend=False,
        ax=axes[2],
    )
    axes[2].set_title(
        f"Satisfação Geral (Pesquisa/Estudo)\n(Mann-Whitney p = {res_mwu_sat.pvalue:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].set_xlabel("Nível Educacional", fontsize=10)
    axes[2].set_ylabel("Satisfaction Score (0 a 10)", fontsize=10)

    plt.tight_layout()
    if salvar_grafico:
        fig.savefig(salvar_grafico, bbox_inches="tight")

    metricas = {
        "Tabela_Proporcao_Percentual": tabela_prop,
        "Chi2_pvalue": round(float(res_qui2.pvalue), 4),
        "MannWhitney_Accuracy_pvalue": round(float(res_mwu_acc.pvalue), 4),
        "MannWhitney_Satisfaction_pvalue": round(float(res_mwu_sat.pvalue), 4),
    }
    return metricas, fig


def analisar_pergunta_9(
    df: pd.DataFrame, salvar_grafico: Optional[str] = None
) -> Tuple[pd.DataFrame, plt.Figure]:
    """Pergunta 9: Impacto no Atendimento e Rotinas Corporativas.

    Compara horas semanais economizadas (Time_Saved_Hours_Per_Week) e
    pontuação de produtividade (Productivity_Score) em Customer Support e Office Work.
    Gráfico: Barras comparativas com rótulos de médias.
    """
    dados = df[df["AI_Purpose"].isin(["Customer Support", "Office Work"])].dropna(
        subset=["Time_Saved_Hours_Per_Week", "Productivity_Score"]
    ).copy()

    resumo = (
        dados.groupby("AI_Purpose")[
            ["Time_Saved_Hours_Per_Week", "Productivity_Score"]
        ]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )

    medias_tempo = dados.groupby("AI_Purpose")["Time_Saved_Hours_Per_Week"].mean()
    medias_prod = dados.groupby("AI_Purpose")["Productivity_Score"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Gráfico 9.1: Horas economizadas
    bars1 = axes[0].bar(
        medias_tempo.index,
        medias_tempo.values,
        color=["#1f77b4", "#ff7f0e"],
        edgecolor="black",
        linewidth=0.6,
        width=0.5,
    )
    axes[0].bar_label(bars1, fmt="%.2f h/sem", padding=3, fontweight="bold")
    axes[0].set_title(
        "Média de Horas Semanais Economizadas", fontsize=13, fontweight="bold"
    )
    axes[0].set_xlabel("Setor / Atividade", fontsize=11)
    axes[0].set_ylabel("Horas Economizadas por Semana", fontsize=11)
    axes[0].set_ylim(0, max(medias_tempo.values) * 1.25)

    # Gráfico 9.2: Produtividade
    bars2 = axes[1].bar(
        medias_prod.index,
        medias_prod.values,
        color=["#2ca02c", "#17becf"],
        edgecolor="black",
        linewidth=0.6,
        width=0.5,
    )
    axes[1].bar_label(bars2, fmt="%.2f", padding=3, fontweight="bold")
    axes[1].set_title(
        "Média da Pontuação de Produtividade Percebida",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xlabel("Setor / Atividade", fontsize=11)
    axes[1].set_ylabel("Productivity Score (0 a 10)", fontsize=11)
    axes[1].set_ylim(0, 10)

    plt.tight_layout()
    if salvar_grafico:
        fig.savefig(salvar_grafico, bbox_inches="tight")
    return resumo, fig


def analisar_pergunta_10(
    df: pd.DataFrame, salvar_grafico: Optional[str] = None
) -> Tuple[Dict[str, Any], plt.Figure]:
    """Pergunta 10: Faixa Etária e Padrão de Adoção.

    10A: Associação entre Faixa Etária e Escolha de Ferramenta de IA (Heatmap e Qui-Quadrado).
    10B: Relação entre Idade e Satisfação com a Solução (Regplot e Correlação de Spearman).
    Gráfico: Painel com 2 subplots integrados.
    """
    df_idade = df.dropna(subset=["Age"]).copy()

    # Criação de faixas etárias
    df_idade["Faixa_Etaria"] = pd.cut(
        df_idade["Age"],
        bins=[15, 24, 34, 120],
        labels=["16-24", "25-34", "35+"],
        include_lowest=True,
    )

    # Agrupamento das principais ferramentas
    principais_ferramentas = ["ChatGPT", "Google Gemini", "Microsoft Copilot"]
    df_idade["Ferramenta_Grupo"] = df_idade["AI_Tool"].where(
        df_idade["AI_Tool"].isin(principais_ferramentas), "Outras Ferramentas"
    )

    # 10A: Tabela de contingência e teste Qui-Quadrado
    tabela_cont = pd.crosstab(df_idade["Faixa_Etaria"], df_idade["Ferramenta_Grupo"])
    tabela_percent = (
        tabela_cont.div(tabela_cont.sum(axis=1), axis=0) * 100
    ).round(1)
    res_qui2 = chi2_contingency(tabela_cont)

    # 10B: Correlação não-paramétrica de Spearman entre Idade e Satisfação
    dados_sat = df.dropna(subset=["Age", "Satisfaction_Score"]).copy()
    res_spearman = spearmanr(dados_sat["Age"], dados_sat["Satisfaction_Score"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 10A: Heatmap de percentual de ferramentas por faixa etária
    sns.heatmap(
        tabela_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "% na Faixa Etária"},
        linewidths=0.5,
        ax=axes[0],
    )
    axes[0].set_title(
        f"Distribuição de Ferramentas por Faixa Etária (%)\n(Qui-Quadrado: p = {res_qui2.pvalue:.3f})",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].set_xlabel("Ferramenta de IA", fontsize=11)
    axes[0].set_ylabel("Faixa Etária", fontsize=11)

    # Gráfico 10B: Regressão de Idade vs Satisfação
    sns.regplot(
        data=dados_sat,
        x="Age",
        y="Satisfaction_Score",
        scatter_kws={"alpha": 0.5, "color": "#1f77b4"},
        line_kws={"color": "#d62728", "linewidth": 2},
        ax=axes[1],
    )
    axes[1].set_title(
        f"Idade vs. Satisfação com IA\n(Spearman: rho = {res_spearman.statistic:.3f}, p = {res_spearman.pvalue:.3f})",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xlabel("Idade (anos)", fontsize=11)
    axes[1].set_ylabel("Satisfaction Score (0 a 10)", fontsize=11)

    plt.tight_layout()
    if salvar_grafico:
        fig.savefig(salvar_grafico, bbox_inches="tight")

    metricas = {
        "Percentual_Ferramentas_Idade": tabela_percent,
        "Chi2_pvalue": round(float(res_qui2.pvalue), 4),
        "Spearman_rho": round(float(res_spearman.statistic), 4),
        "Spearman_pvalue": round(float(res_spearman.pvalue), 4),
    }
    return metricas, fig
