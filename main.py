"""
Projeto 1 - C11: Ciência de Dados (Inatel)
Análise de Adoção e Impacto de IA em Estudantes e Profissionais

Execução principal das análises e resumo executivo das Perguntas 1 a 5.
"""

import os
from src.analises import (
    carregar_e_limpar_dados,
    analisar_pergunta_1,
    analisar_pergunta_2,
    analisar_pergunta_3,
    analisar_pergunta_4,
    analisar_pergunta_5,
)


def main():
    print("=" * 80)
    print("PROJETO 1 - C11: CIÊNCIA DE DADOS (INATEL)")
    print("Análise de Adoção e Impacto de IA em Estudantes e Profissionais")
    print("=" * 80)

    # 1. Carga e higienização dos dados
    print("\n[ETAPA 1] Carregando e higienizando os dados...")
    df = carregar_e_limpar_dados()
    print(f"✔ Dataset carregado com sucesso!")
    print(f"  • Total de registros após deduplicação e limpeza: {len(df)}")
    print(f"  • Total de atributos processados: {len(df.columns)}")

    # 2. Pergunta 1
    print("\n" + "-" * 80)
    print("PERGUNTA 1: Estudantes vs. Profissionais (Uso diário e Produtividade)")
    print("-" * 80)
    resumo_p1, _ = analisar_pergunta_1(df)
    print(resumo_p1)

    # 3. Pergunta 2
    print("\n" + "-" * 80)
    print("PERGUNTA 2: Preferência de Ferramentas de IA")
    print("-" * 80)
    resumo_p2, _ = analisar_pergunta_2(df)
    print(resumo_p2.head(8))

    # 4. Pergunta 3
    print("\n" + "-" * 80)
    print("PERGUNTA 3: Tempo de Uso vs. Economia Real (Correlação e Saturação)")
    print("-" * 80)
    metricas_p3, _ = analisar_pergunta_3(df)
    print(f"  • Coeficiente de Pearson (r): {metricas_p3['Pearson_r']}")
    print(f"  • Coeficiente de Spearman (r): {metricas_p3['Spearman_r']}")
    print(f"  • Coeficientes do Polinômio grau 2: {metricas_p3['Coeficientes_Polinomio']}")

    # 5. Pergunta 4
    print("\n" + "-" * 80)
    print("PERGUNTA 4: Propósito de Uso vs. Satisfação e Acurácia")
    print("-" * 80)
    resumo_p4, _ = analisar_pergunta_4(df)
    print(resumo_p4)

    # 6. Pergunta 5
    print("\n" + "-" * 80)
    print("PERGUNTA 5: Investimento Financeiro (Planos Pagos vs. Gratuitos)")
    print("-" * 80)
    resumo_p5, _ = analisar_pergunta_5(df)
    print(resumo_p5)

    print("\n" + "=" * 80)
    print("✔ Pipeline das Perguntas 1 a 5 concluído com sucesso!")
    print("📌 Para ver os gráficos completos e a continuidade das Perguntas 6 a 10,")
    print("   abra o notebook interativo:")
    print("   jupyter notebook notebooks/analise_projeto_c11.ipynb")
    print("=" * 80)


if __name__ == "__main__":
    main()