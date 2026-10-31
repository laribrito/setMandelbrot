#!/usr/bin/env python3
"""
Geração do CSV Geral Consolidado para o Relatório MPI (Mandelbrot)
-----------------------------------------------------------------
Lê os dados tratados de data/dados_tratados_mpi.csv e gera o arquivo
consolidado único data/tabela_geral_mpi.csv.

Utiliza como referência T1 canônica o algoritmo: ponto_a_ponto_v2
Exibe explicitamente:
  - As linhas do baseline ponto_a_ponto_v2 no topo de cada cenário
  - A coluna 'Tempo T1 (s)' ao lado de 'Tempo Tp (s)' em todas as linhas
  - As duas eficiências (Ep=4 e Efísico) para testes com 4 núcleos

Colunas geradas:
  - Cenário
  - Estratégia
  - Resolução (ex: 4096 × 4096 (4K))
  - Chunk (ex: N/p para bloco, 1, 2, 4 ou — para sequencial)
  - Tempo T1 (s) (Tempo de referência do ponto_a_ponto_v2)
  - Tempo Tp (s) (Tempo paralelo tratado)
  - Speedup (Sp) (Sp = T1 / Tp)
  - Ep=4 (Sp / 4)
  - Efísico (Sp / 2, núcleos físicos)
  - T1_s, Tempo_s, Speedup, Ep=4_%, Efísico_% (valores numéricos float)
  - Std_s, CV_% (métricas de dispersão)
  - Code, Schedule, ChunkSize
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


def format_chunk(schedule: str, chunk_size: int, code: str) -> str:
    """Formata representação do chunk: 'N/p' para bloco, '—' para sequencial."""
    if "ponto_a_ponto" in code or schedule == "sequential":
        return "—"
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
    if code == "ponto_a_ponto_v2":
        return "Referência Sequencial (ponto_a_ponto_v2)"
    elif code == "ponto_a_ponto":
        return "Sequencial Original (ponto_a_ponto)"
    elif code == "mpi_estatico" and schedule == "block":
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


def gerar_csv_consolidado(
    input_path: str,
    output_csv_path: str,
    threads: int = 4,
    physical_cores: int = 2,
    csv_delimiter: str = ","
) -> pd.DataFrame:
    """Gera o CSV geral consolidado com dados de MPI e o baseline ponto_a_ponto_v2."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo de dados tratados não encontrado: '{input_path}'")

    df = pd.read_csv(input_path)

    # 1. Filtra execuções MPI com a quantidade de threads/processos desejada (p=4)
    #    E também as 6 linhas de baseline canônico (ponto_a_ponto_v2)
    filtro = (df["Code"] == "ponto_a_ponto_v2") | (
        df["Code"].str.startswith("mpi_") & (df["Threads"] == threads)
    )
    df_filtrado = df[filtro].copy()

    if df_filtrado.empty:
        print(f"⚠️ Nenhuma execução encontrada em '{input_path}' para o filtro aplicado.")
        return pd.DataFrame()

    # 2. Mapear estratégia legível
    df_filtrado["Estratégia"] = df_filtrado.apply(
        lambda r: map_strategy_name(r["Code"], r["Schedule"]), axis=1
    )

    # 3. Eficiência física: usa do arquivo se existir ou calcula Sp / physical_cores
    if "Eficiencia_Fisica" in df_filtrado.columns:
        df_filtrado["Eficiencia_Fisica"] = df_filtrado["Eficiencia_Fisica"].fillna(
            df_filtrado["Speedup_Tratado"] / float(physical_cores)
        )
    else:
        df_filtrado["Eficiencia_Fisica"] = df_filtrado.apply(
            lambda r: 1.0 if r["Code"] == "ponto_a_ponto_v2" else r["Speedup_Tratado"] / float(physical_cores),
            axis=1
        )

    # 4. Ordenação canônica do relatório:
    #    Cenário (Visão Completa -> Zoom)
    #    Estratégia (ponto_a_ponto_v2 no topo como referência, depois Estático Bloco, Cíclico, Dinâmico...)
    ordem_estrategias = {
        "Referência Sequencial (ponto_a_ponto_v2)": 0,
        "Estático Bloco": 1,
        "Estático Cíclico": 2,
        "Dinâmico Dedicado": 3,
        "Dinâmico Participativo": 4,
        "Dinâmico Meia-Linha": 5
    }
    ordem_cenarios = {
        "Visão Completa": 0,
        "Zoom (Seahorse)": 1
    }

    df_filtrado["_ord_scen"] = df_filtrado["Scenario"].map(lambda s: ordem_cenarios.get(s, 99))
    df_filtrado["_ord_strat"] = df_filtrado["Estratégia"].map(lambda e: ordem_estrategias.get(e, 99))
    df_filtrado.sort_values(
        by=["_ord_scen", "_ord_strat", "WIDTH", "ChunkSize"],
        inplace=True
    )
    df_filtrado.drop(columns=["_ord_scen", "_ord_strat"], inplace=True)

    # 5. Colunas formatadas (padrão com vírgula)
    df_filtrado["Cenário"] = df_filtrado["Scenario"]
    df_filtrado["Resolução"] = df_filtrado["WIDTH"].apply(format_resolution)
    df_filtrado["Chunk"] = df_filtrado.apply(
        lambda r: format_chunk(r["Schedule"], r["ChunkSize"], r["Code"]), axis=1
    )

    df_filtrado["Tempo T1 (s)"] = df_filtrado["T1_Tratado"].apply(lambda x: format_decimal(x, 2))
    df_filtrado["Tempo Tp (s)"] = df_filtrado["Tempo_Tratado"].apply(lambda x: format_decimal(x, 2))
    df_filtrado["Speedup (Sp)"] = df_filtrado["Speedup_Tratado"].apply(lambda x: f"{format_decimal(x, 2)}×")
    df_filtrado["Ep=4"] = df_filtrado["Eficiencia_Tratada"].apply(lambda x: f"{format_decimal(x * 100.0, 1)}%")
    df_filtrado["Efísico"] = df_filtrado["Eficiencia_Fisica"].apply(lambda x: f"{format_decimal(x * 100.0, 1)}%")

    # Métricas numéricas puras
    df_filtrado["T1_s"] = df_filtrado["T1_Tratado"].round(4)
    df_filtrado["Tempo_s"] = df_filtrado["Tempo_Tratado"].round(4)
    df_filtrado["Speedup"] = df_filtrado["Speedup_Tratado"].round(4)
    df_filtrado["Ep=4_%"] = (df_filtrado["Eficiencia_Tratada"] * 100.0).round(2)
    df_filtrado["Efísico_%"] = (df_filtrado["Eficiencia_Fisica"] * 100.0).round(2)
    df_filtrado["Std_s"] = df_filtrado["Std_Tratado"].round(4)
    df_filtrado["CV_%"] = df_filtrado["CV_Tratado_%"].round(2)

    # 6. Seleção final das colunas
    colunas_finais = [
        "Cenário",
        "Estratégia",
        "Resolução",
        "Chunk",
        "Tempo T1 (s)",
        "Tempo Tp (s)",
        "Speedup (Sp)",
        "Ep=4",
        "Efísico",
        "T1_s",
        "Tempo_s",
        "Speedup",
        "Ep=4_%",
        "Efísico_%",
        "Std_s",
        "CV_%",
        "Code",
        "Schedule",
        "ChunkSize"
    ]

    df_saida = df_filtrado[colunas_finais].copy()

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_saida.to_csv(output_csv_path, index=False, sep=csv_delimiter, encoding="utf-8")

    n_base = (df_saida["Code"] == "ponto_a_ponto_v2").sum()
    n_mpi = (df_saida["Code"].str.startswith("mpi_")).sum()

    print(f"✅ CSV Geral Consolidado salvo com sucesso em: {output_csv_path}")
    print(f"📊 Total de registros: {len(df_saida)} ({n_base} referências ponto_a_ponto_v2 + {n_mpi} MPI p={threads})\n")

    # Exibe amostra das primeiras linhas com a referência e o Estático Bloco
    cols_previa = ["Cenário", "Estratégia", "Resolução", "Chunk", "Tempo T1 (s)", "Tempo Tp (s)", "Speedup (Sp)", "Ep=4", "Efísico"]
    print("Amostra inicial (Baseline ponto_a_ponto_v2 + Estático Bloco na Visão Completa):")
    print(df_saida[cols_previa].head(6).to_string(index=False))
    print()

    return df_saida


def main():
    parser = argparse.ArgumentParser(
        description="Gera o CSV geral consolidado com dados de benchmark MPI e baseline ponto_a_ponto_v2."
    )
    parser.add_argument(
        "--input",
        default=resolve_path("data/dados_tratados_mpi.csv"),
        help="Caminho do CSV de dados tratados (padrão: data/dados_tratados_mpi.csv)"
    )
    parser.add_argument(
        "--output",
        default=resolve_path("data/tabela_geral_mpi.csv"),
        help="Caminho do CSV consolidado de saída (padrão: data/tabela_geral_mpi.csv)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Quantidade de processos MPI a filtrar (padrão: 4)"
    )
    parser.add_argument(
        "--physical-cores",
        type=int,
        default=2,
        help="Quantidade de núcleos físicos para Efísico (padrão: 2)"
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="Delimitador do CSV de saída (padrão: ',' ; use ';' para Excel em português)"
    )

    args = parser.parse_args()

    gerar_csv_consolidado(
        input_path=args.input,
        output_csv_path=args.output,
        threads=args.threads,
        physical_cores=args.physical_cores,
        csv_delimiter=args.sep
    )


if __name__ == "__main__":
    main()
