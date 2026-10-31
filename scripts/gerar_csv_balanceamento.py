#!/usr/bin/env python3
"""
Geração de CSV para Análise de Balanceamento de Carga MPI (Mandelbrot)
--------------------------------------------------------------------
Consolida os tempos individuais de cada processo (Rank 0 a Rank p-1)
e calcula o Fator de Balanço de Carga (F_LB) conforme a Equação 2.4
do relatório de Metodologia:

    F_LB = (T_max - T_min) / T_max

Valores próximos a 0,0 indicam condição de ótimo balanceamento temporal,
enquanto valores próximos a 1,0 indicam forte desbalanceamento com espera excessiva.

Gera o arquivo:
  data/analise_balanceamento_carga_mpi.csv

Colunas:
  - Cenário, Estratégia, Resolução, Chunk
  - T_R0 (s), T_R1 (s), T_R2 (s), T_R3 (s) (tempos individuais de cada rank)
  - T_max (s), T_min (s), Delta_T (s)
  - F_LB, F_LB (%) (Fator de Balanço Global - Eq. 2.4)
  - F_LB_Workers, F_LB_Workers (%) (Fator entre trabalhadores R1 a R3)
  - Media_Workers (s), Desvio_Workers (s)
  - Colunas numéricas brutas (float) para uso programático e gráficos
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np


def resolve_path(rel_path: str) -> str:
    """Resolve caminho relativo a partir da raiz do repositório."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    return os.path.normpath(os.path.join(root_dir, rel_path))


def format_resolution(width: int) -> str:
    """Formata resolução em texto legível."""
    k_map = {4096: "4K", 8192: "8K", 16384: "16K"}
    k_str = k_map.get(width, f"{width // 1024}K")
    return f"{width} × {width} ({k_str})"


def format_chunk(schedule: str, chunk_size: int) -> str:
    """Formata representação do chunk: 'N/p' para divisão em bloco contíguo."""
    if schedule == "block" or chunk_size == 0:
        return "N/p"
    return str(int(chunk_size))


def format_decimal(value: float, decimals: int = 2) -> str:
    """Formata número com vírgula decimal (padrão brasileiro / ABNT)."""
    if pd.isna(value):
        return "N/A"
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(value).replace(".", ",")


def map_strategy_name(code: str, schedule: str) -> str:
    """Mapeia código e schedule para nome legível da estratégia."""
    if code == "mpi_estatico" and schedule == "block":
        return "Estático Bloco"
    elif code == "mpi_estatico" and schedule == "cyclic":
        return "Estático Cíclico"
    elif code == "mpi_dinamico":
        return "Dinâmico Dedicado"
    elif code == "mpi_dinamico_participativo":
        return "Dinâmico Participativo"
    elif code == "mpi_dinamico_meia_linha":
        return "Dinâmico Meia-Linha"
    return f"{code} ({schedule})"


def gerar_csv_balanceamento(
    input_path: str,
    output_csv_path: str,
    threads: int = 4,
    csv_delimiter: str = ","
) -> pd.DataFrame:
    """Consolida os dados de tempos por rank e calcula F_LB e métricas de balanceamento."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo de dados de ranks não encontrado: '{input_path}'")

    df = pd.read_csv(input_path)

    # Filtrar execuções MPI com a quantidade de processos desejada (p=4)
    mpi_df = df[df["Code"].str.startswith("mpi_") & (df["Threads"] == threads)].copy()

    if mpi_df.empty:
        print(f"⚠️ Nenhuma execução MPI encontrada em '{input_path}' para Threads={threads}.")
        return pd.DataFrame()

    # Mapear estratégia legível
    mpi_df["Estratégia"] = mpi_df.apply(lambda r: map_strategy_name(r["Code"], r["Schedule"]), axis=1)

    # Pivotar tempos por rank: transformar Ranks 0, 1, 2, 3 em colunas
    piv = mpi_df.pivot(
        index=["Scenario", "Estratégia", "Code", "Schedule", "ChunkSize", "WIDTH"],
        columns="Rank",
        values="Tempo_Rank_Tratado"
    ).reset_index()
    piv.columns.name = None

    # Renomear colunas de ranks
    rank_cols = [c for c in piv.columns if isinstance(c, (int, np.integer)) or str(c).isdigit()]
    rank_rename = {r: f"T_R{r}" for r in rank_cols}
    piv.rename(columns=rank_rename, inplace=True)
    cols_t_ranks = [f"T_R{r}" for r in sorted(rank_cols)]

    # Cálculo dos extremos e métricas globais (todos os ranks)
    piv["T_max"] = piv[cols_t_ranks].max(axis=1)
    piv["T_min"] = piv[cols_t_ranks].min(axis=1)
    piv["Delta_T"] = piv["T_max"] - piv["T_min"]

    # Fator de Balanço de Carga Global: F_LB = (T_max - T_min) / T_max (Equação 2.4)
    piv["F_LB"] = np.where(piv["T_max"] > 0, piv["Delta_T"] / piv["T_max"], 0.0)
    piv["F_LB_%"] = piv["F_LB"] * 100.0

    # Cálculo específico para trabalhadores (Ranks >= 1)
    # Fundamental para Coordenador Dedicado (onde R0 não computa pixels)
    worker_cols = [c for c in cols_t_ranks if c != "T_R0"]
    if worker_cols:
        piv["T_max_workers"] = piv[worker_cols].max(axis=1)
        piv["T_min_workers"] = piv[worker_cols].min(axis=1)
        piv["Delta_T_workers"] = piv["T_max_workers"] - piv["T_min_workers"]
        piv["F_LB_workers"] = np.where(
            piv["T_max_workers"] > 0,
            piv["Delta_T_workers"] / piv["T_max_workers"],
            0.0
        )
        piv["F_LB_workers_%"] = piv["F_LB_workers"] * 100.0
        piv["Media_Workers"] = piv[worker_cols].mean(axis=1)
        piv["Desvio_Workers"] = piv[worker_cols].std(axis=1, ddof=1).fillna(0.0)
    else:
        piv["T_max_workers"] = piv["T_max"]
        piv["T_min_workers"] = piv["T_min"]
        piv["Delta_T_workers"] = piv["Delta_T"]
        piv["F_LB_workers"] = piv["F_LB"]
        piv["F_LB_workers_%"] = piv["F_LB_%"]
        piv["Media_Workers"] = piv["T_max"]
        piv["Desvio_Workers"] = 0.0

    # Ordenação canônica do relatório
    ordem_estrategias = {
        "Estático Bloco": 0,
        "Estático Cíclico": 1,
        "Dinâmico Dedicado": 2,
        "Dinâmico Participativo": 3,
        "Dinâmico Meia-Linha": 4
    }
    ordem_cenarios = {
        "Visão Completa": 0,
        "Zoom (Seahorse)": 1
    }

    piv["_ord_scen"] = piv["Scenario"].map(lambda s: ordem_cenarios.get(s, 99))
    piv["_ord_strat"] = piv["Estratégia"].map(lambda e: ordem_estrategias.get(e, 99))
    piv.sort_values(
        by=["_ord_scen", "_ord_strat", "WIDTH", "ChunkSize"],
        inplace=True
    )
    piv.drop(columns=["_ord_scen", "_ord_strat"], inplace=True)

    # Formatações com vírgula (ABNT)
    piv["Cenário"] = piv["Scenario"]
    piv["Resolução"] = piv["WIDTH"].apply(format_resolution)
    piv["Chunk"] = piv.apply(lambda r: format_chunk(r["Schedule"], r["ChunkSize"]), axis=1)

    for c in cols_t_ranks:
        piv[f"{c} (s)"] = piv[c].apply(lambda x: format_decimal(x, 2))

    piv["T_max (s)"] = piv["T_max"].apply(lambda x: format_decimal(x, 2))
    piv["T_min (s)"] = piv["T_min"].apply(lambda x: format_decimal(x, 2))
    piv["Delta_T (s)"] = piv["Delta_T"].apply(lambda x: format_decimal(x, 2))
    piv["Fator F_LB"] = piv["F_LB"].apply(lambda x: format_decimal(x, 4))
    piv["F_LB (%)"] = piv["F_LB_%"].apply(lambda x: f"{format_decimal(x, 1)}%")

    piv["F_LB_Workers"] = piv["F_LB_workers"].apply(lambda x: format_decimal(x, 4))
    piv["F_LB_Workers (%)"] = piv["F_LB_workers_%"].apply(lambda x: f"{format_decimal(x, 1)}%")
    piv["Media_Workers (s)"] = piv["Media_Workers"].apply(lambda x: format_decimal(x, 2))
    piv["Desvio_Workers (s)"] = piv["Desvio_Workers"].apply(lambda x: format_decimal(x, 4))

    # Valores numéricos puros (float arredondados)
    piv["T_max_num"] = piv["T_max"].round(4)
    piv["T_min_num"] = piv["T_min"].round(4)
    piv["Delta_T_num"] = piv["Delta_T"].round(4)
    piv["F_LB_num"] = piv["F_LB"].round(4)
    piv["F_LB_workers_num"] = piv["F_LB_workers"].round(4)
    piv["Media_Workers_num"] = piv["Media_Workers"].round(4)
    piv["Desvio_Workers_num"] = piv["Desvio_Workers"].round(4)

    # Montagem das colunas finais
    colunas_finais = [
        "Cenário",
        "Estratégia",
        "Resolução",
        "Chunk",
        "T_R0 (s)", "T_R1 (s)", "T_R2 (s)", "T_R3 (s)",
        "T_max (s)", "T_min (s)", "Delta_T (s)",
        "Fator F_LB", "F_LB (%)",
        "F_LB_Workers", "F_LB_Workers (%)",
        "Media_Workers (s)", "Desvio_Workers (s)",
        "T_max_num", "T_min_num", "Delta_T_num",
        "F_LB_num", "F_LB_workers_num",
        "Media_Workers_num", "Desvio_Workers_num",
        "Code", "Schedule", "ChunkSize", "WIDTH"
    ]

    df_saida = piv[[c for c in colunas_finais if c in piv.columns]].copy()

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_saida.to_csv(output_csv_path, index=False, sep=csv_delimiter, encoding="utf-8")

    print(f"✅ CSV de Balanceamento de Carga salvo com sucesso em: {output_csv_path}")
    print(f"📊 Total de configurações analisadas: {len(df_saida)} (p={threads})\n")

    # Exibir amostra comparando Bloco vs Cíclico vs Dinâmico no Zoom Seahorse 8K
    print("=" * 115)
    print("⚖️  COMPARATIVO DE FATOR DE BALANÇO DE CARGA (F_LB - Eq. 2.4) — ZOOM SEAHORSE 8K:")
    print("=" * 115)
    cols_amostra = [
        "Estratégia", "Chunk", "T_R0 (s)", "T_R1 (s)", "T_R2 (s)", "T_R3 (s)",
        "T_max (s)", "T_min (s)", "Delta_T (s)", "F_LB (%)", "F_LB_Workers (%)"
    ]
    amostra = df_saida[
        (df_saida["Cenário"] == "Zoom (Seahorse)") &
        (df_saida["Resolução"].str.contains("8192")) &
        (df_saida["Chunk"].isin(["N/p", "1"]))
    ]
    print(amostra[cols_amostra].to_string(index=False))
    print("=" * 115)
    print("💡 Interpretação:")
    print("   • Estático Bloco: F_LB = 45,6% (R2 demorou 11,77s enquanto R1 terminou em 6,40s; espera de 5,36s).")
    print("   • Estático Cíclico (ch=1): F_LB cai para 6,3% (distribuição intercalada equilibra as faixas densas).")
    print("   • Dinâmico Dedicado (ch=1): F_LB entre trabalhadores = 0,2% (equilíbrio quase perfeito entre R1, R2 e R3).\n")

    return df_saida


def main():
    parser = argparse.ArgumentParser(
        description="Gera CSV com dados para análise de balanceamento de carga MPI (Fator F_LB - Eq. 2.4)."
    )
    parser.add_argument(
        "--input",
        default=resolve_path("data/dados_tratados_ranks_mpi.csv"),
        help="Caminho do CSV de ranks tratados (padrão: data/dados_tratados_ranks_mpi.csv)"
    )
    parser.add_argument(
        "--output",
        default=resolve_path("data/analise_balanceamento_carga_mpi.csv"),
        help="Caminho do CSV de saída (padrão: data/analise_balanceamento_carga_mpi.csv)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Quantidade de processos a analisar (padrão: 4)"
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="Delimitador do CSV de saída (padrão: ',' ; use ';' para Excel em português)"
    )

    args = parser.parse_args()

    gerar_csv_balanceamento(
        input_path=args.input,
        output_csv_path=args.output,
        threads=args.threads,
        csv_delimiter=args.sep
    )


if __name__ == "__main__":
    main()
