"""
Projeto 1 - C11: Ciência de Dados (Inatel)
Análise de Adoção e Impacto de IA em Estudantes e Profissionais

Execução principal das análises e resumo executivo das 10 Perguntas de Pesquisa.
"""

import os
from src.analises import (
    carregar_e_limpar_dados,
    analisar_pergunta_1,
    analisar_pergunta_2,
    analisar_pergunta_3,
    analisar_pergunta_4,
    analisar_pergunta_5,
    analisar_pergunta_6,
    analisar_pergunta_7,
    analisar_pergunta_8,
    analisar_pergunta_9,
    analisar_pergunta_10,
)


def main():
    print("=" * 80)
    print("PROJETO 1 - C11: CIÊNCIA DE DADOS (INATEL)")
    print("Análise de Adoção e Impacto de IA em Estudantes e Profissionais")
    print("=" * 80)

    # Diretório para exportação de gráficos
    pasta_graficos = os.path.join("notebooks", "graficos")
    os.makedirs(pasta_graficos, exist_ok=True)

    # 1. Carga e higienização dos dados
    print("\n[ETAPA 1] Carregando e higienizando os dados...")
    df = carregar_e_limpar_dados()
    print("Dataset carregado com sucesso!")
    print(f"  • Total de registros após deduplicação e limpeza: {len(df)}")
    print(f"  • Total de atributos processados: {len(df.columns)}")

    # 2. Pergunta 1
    print("\n" + "-" * 80)
    print("PERGUNTA 1: Estudantes vs. Profissionais (Uso diário e Produtividade)")
    print("-" * 80)
    resumo_p1, _ = analisar_pergunta_1(
        df, salvar_grafico=os.path.join(pasta_graficos, "pergunta_1.png")
    )
    print(resumo_p1)

    # 3. Pergunta 2
    print("\n" + "-" * 80)
    print("PERGUNTA 2: Preferência de Ferramentas de IA")
    print("-" * 80)
    resumo_p2, _ = analisar_pergunta_2(
        df, salvar_grafico=os.path.join(pasta_graficos, "pergunta_2.png")
    )
    print(resumo_p2.head(8))

    # 4. Pergunta 3
    print("\n" + "-" * 80)
    print("PERGUNTA 3: Tempo de Uso vs. Economia Real (Correlação e Saturação)")
    print("-" * 80)
    metricas_p3, _ = analisar_pergunta_3(
        df, salvar_grafico=os.path.join(pasta_graficos, "pergunta_3.png")
    )
    print(f"  • Coeficiente de Pearson (r): {metricas_p3['Pearson_r']}")
    print(f"  • Coeficiente de Spearman (r): {metricas_p3['Spearman_r']}")
    print(f"  • Coeficientes do Polinômio grau 2: {metricas_p3['Coeficientes_Polinomio']}")

    # 5. Pergunta 4
    print("\n" + "-" * 80)
    print("PERGUNTA 4: Propósito de Uso vs. Satisfação e Acurácia")
    print("-" * 80)
    resumo_p4, _ = analisar_pergunta_4(
        df, salvar_grafico=os.path.join(pasta_graficos, "pergunta_4.png")
    )
    print(resumo_p4)

    # 6. Pergunta 5
    print("\n" + "-" * 80)
    print("PERGUNTA 5: Investimento Financeiro (Planos Pagos vs. Gratuitos)")
    print("-" * 80)
    resumo_p5, _ = analisar_pergunta_5(
        df, salvar_grafico=os.path.join(pasta_graficos, "pergunta_5.png")
    )
    print(resumo_p5)

    # 7. Pergunta 6
    print("\n" + "-" * 80)
    print("PERGUNTA 6: Acurácia Percebida por Ferramenta (Teste de Kruskal-Wallis)")
    print("-" * 80)
    resumo_p6, _ = analisar_pergunta_6(
        df, salvar_grafico=os.path.join(pasta_graficos, "pergunta_6.png")
    )
    print(resumo_p6)
    p6_val = resumo_p6["Valor_p"].iloc[0]
    p6_concl = (
        "Diferença estatisticamente significativa encontrada."
        if p6_val < 0.05
        else "Nenhuma diferença estatisticamente significativa entre as ferramentas."
    )
    print(f"  • Conclusão Estatística (p = {p6_val}): {p6_concl}")

    # 8. Pergunta 7
    print("\n" + "-" * 80)
    print("PERGUNTA 7: Renda Mensal: Tecnologia/Dados vs. Suporte (Teste Mann-Whitney U)")
    print("-" * 80)
    resumo_p7, _ = analisar_pergunta_7(
        df, salvar_grafico=os.path.join(pasta_graficos, "pergunta_7.png")
    )
    print(resumo_p7)
    p7_val = resumo_p7["Valor_p"].iloc[0]
    p7_concl = (
        "Diferença de renda significativa."
        if p7_val < 0.05
        else "Sem diferença estatisticamente significativa nas rendas."
    )
    print(f"  • Conclusão Estatística (p = {p7_val}): {p7_concl}")

    # 9. Pergunta 8
    print("\n" + "-" * 80)
    print("PERGUNTA 8: Nível Educacional e Adoção para Pesquisa (Qui-Quadrado & Mann-Whitney)")
    print("-" * 80)
    metricas_p8, _ = analisar_pergunta_8(
        df, salvar_grafico=os.path.join(pasta_graficos, "pergunta_8.png")
    )
    print("  • Distribuição percentual de uso por finalidade:")
    print(metricas_p8["Tabela_Proporcao_Percentual"][["Research", "Studying"]])
    print(f"  • Valor-p Qui-Quadrado (Uso Pesquisa/Estudo): {metricas_p8['Chi2_pvalue']}")
    print(f"  • Valor-p Mann-Whitney (Acurácia Percebida): {metricas_p8['MannWhitney_Accuracy_pvalue']}")
    print(f"  • Valor-p Mann-Whitney (Satisfação Geral): {metricas_p8['MannWhitney_Satisfaction_pvalue']}")

    # 10. Pergunta 9
    print("\n" + "-" * 80)
    print("PERGUNTA 9: Impacto no Atendimento e Rotinas Corporativas")
    print("-" * 80)
    resumo_p9, _ = analisar_pergunta_9(
        df, salvar_grafico=os.path.join(pasta_graficos, "pergunta_9.png")
    )
    print(resumo_p9)

    # 11. Pergunta 10
    print("\n" + "-" * 80)
    print("PERGUNTA 10: Faixa Etária vs. Ferramenta & Satisfação (Qui-Quadrado & Spearman)")
    print("-" * 80)
    metricas_p10, _ = analisar_pergunta_10(
        df, salvar_grafico=os.path.join(pasta_graficos, "pergunta_10.png")
    )
    print("  • Distribuição das ferramentas por faixa etária (%):")
    print(metricas_p10["Percentual_Ferramentas_Idade"])
    print(f"  • Valor-p Qui-Quadrado (Idade x Ferramenta): {metricas_p10['Chi2_pvalue']}")
    print(f"  • Correlação de Spearman (Idade x Satisfação): rho = {metricas_p10['Spearman_rho']}, p = {metricas_p10['Spearman_pvalue']}")

    print("\n" + "=" * 80)
    print("Pipeline completo das 10 Perguntas executado com sucesso!")
    print(f"Todos os 10 gráficos salvos em: {pasta_graficos}/")
    print("Para explorar interativamente os dados e gráficos:")
    print("   jupyter notebook notebooks/analise_projeto_c11.ipynb")
    print("=" * 80)


if __name__ == "__main__":
    main()