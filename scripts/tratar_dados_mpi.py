#!/usr/bin/env python3
"""
Tratamento Estatístico de Dados de Benchmark (Mandelbrot MPI e Ponto a Ponto)
-----------------------------------------------------------------------------
Aplica o método de média aparada/truncada (descarte de min e max de 5 repetições)
aos tempos de execução de:
  1) ponto_a_ponto (Baseline sequencial de referência T1)
  2) Algoritmos MPI (Tempos totais de execução Tp, Speedup e Eficiência)
  3) Tempos individuais de cada processo/rank/thread (Balanceamento de Carga)

Gera:
  - data/dados_tratados_mpi.csv (Tempos globais, Speedup e Eficiência tratados)
  - data/dados_tratados_ranks_mpi.csv (Tempos por rank e métricas de balanceamento)
  - Relatório estruturado e formatado no terminal
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd


# Configuração padrão do pandas para terminal
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def classify_scenario(row: pd.Series) -> str:
    """Classifica o cenário fractal com base nas coordenadas e iterações máximas."""
    re_min = row.get("RE_MIN", 0.0)
    max_iter = row.get("MAX_ITER", 1000)
    if abs(re_min - (-2.0)) < 0.05 and max_iter <= 1500:
        return "Visão Completa"
    elif abs(re_min - (-0.745144)) < 0.01 or max_iter >= 3000:
        return "Zoom (Seahorse)"
    return "Personalizado"


def aplicar_tratativa(tempos: list[float] | np.ndarray) -> dict:
    """
    Aplica a tratativa estatística aos tempos de repetição:
      - N >= 5: considera o lote de 5 execuções (as 5 mais recentes), descarta o menor
        e o maior valor, e calcula média e desvio padrão dos 3 tempos centrais.
      - N in [3, 4]: descarta menor e maior e calcula a média dos restantes.
      - N < 3: média simples.
    """
    arr = np.array(tempos, dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "N_Bruto": 0, "N_Usado": 0, "Tempo_Tratado": np.nan, "Std_Tratado": np.nan,
            "CV_Tratado_%": np.nan, "Min_Descartado": np.nan, "Max_Descartado": np.nan,
            "Tempos_Brutos": []
        }

    # Se houver mais de 5, avalia as 5 execuções mais recentes
    t_eval = arr[-5:] if n >= 5 else arr
    sorted_t = np.sort(t_eval)

    if len(sorted_t) >= 5:
        min_desc = sorted_t[0]
        max_desc = sorted_t[-1]
        mid_vals = sorted_t[1:4]
    elif len(sorted_t) in [3, 4]:
        # Para 3 repetições (OpenMP): preserva os 3 valores para ter estimador BLUE e graus de liberdade para desvio padrão
        min_desc = np.nan
        max_desc = np.nan
        mid_vals = sorted_t
    else:
        min_desc = np.nan
        max_desc = np.nan
        mid_vals = sorted_t

    mean_tratado = float(np.mean(mid_vals))
    std_tratado = float(np.std(mid_vals, ddof=1)) if len(mid_vals) > 1 else 0.0
    cv_tratado = (std_tratado / mean_tratado * 100.0) if mean_tratado > 0 else 0.0

    return {
        "N_Bruto": n,
        "N_Usado": len(mid_vals),
        "Tempo_Tratado": mean_tratado,
        "Std_Tratado": std_tratado,
        "CV_Tratado_%": cv_tratado,
        "Min_Descartado": float(min_desc) if pd.notna(min_desc) else np.nan,
        "Max_Descartado": float(max_desc) if pd.notna(max_desc) else np.nan,
        "Tempos_Brutos": list(np.round(sorted_t, 4))
    }


def carregar_dados(exec_path: str = "data/dateTimeExecution.csv",
                   ranks_path: str = "data/tempos_processos_mpi.csv",
                   machine: str = "all",
                   ignorar_512: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega e padroniza os arquivos CSV de execuções globais e tempos por processo."""
    if not os.path.exists(exec_path):
        print(f"❌ Arquivo de execuções '{exec_path}' não encontrado.")
        sys.exit(1)
    if not os.path.exists(ranks_path):
        print(f"❌ Arquivo de tempos por processo '{ranks_path}' não encontrado.")
        sys.exit(1)

    df_exec = pd.read_csv(exec_path)
    df_ranks = pd.read_csv(ranks_path)

    # Padronização df_exec
    df_exec["TempoGasto"] = pd.to_numeric(df_exec["TempoGasto"], errors="coerce")
    df_exec = df_exec.dropna(subset=["TempoGasto"])
    df_exec["Schedule"] = df_exec["Schedule"].fillna("N/A").astype(str).str.lower()
    df_exec["ChunkSize"] = pd.to_numeric(df_exec["ChunkSize"], errors="coerce").fillna(0).astype(int)
    df_exec["Threads"] = pd.to_numeric(df_exec["Threads"], errors="coerce").fillna(1).astype(int)
    df_exec["WIDTH"] = pd.to_numeric(df_exec["WIDTH"], errors="coerce").fillna(0).astype(int)
    df_exec["HEIGHT"] = pd.to_numeric(df_exec["HEIGHT"], errors="coerce").fillna(0).astype(int)
    df_exec["MAX_ITER"] = pd.to_numeric(df_exec["MAX_ITER"], errors="coerce").fillna(0).astype(int)
    df_exec["Machine"] = df_exec["Machine"].fillna("Unknown").astype(str)
    df_exec["Code"] = df_exec["Code"].astype(str)
    df_exec["Scenario"] = df_exec.apply(classify_scenario, axis=1)

    # Padronização df_ranks
    df_ranks["TempoProcesso"] = pd.to_numeric(df_ranks["TempoProcesso"], errors="coerce")
    df_ranks = df_ranks.dropna(subset=["TempoProcesso"])
    df_ranks["Schedule"] = df_ranks["Schedule"].fillna("N/A").astype(str).str.lower()
    df_ranks["ChunkSize"] = pd.to_numeric(df_ranks["ChunkSize"], errors="coerce").fillna(0).astype(int)
    df_ranks["Threads"] = pd.to_numeric(df_ranks["Threads"], errors="coerce").fillna(1).astype(int)
    df_ranks["WIDTH"] = pd.to_numeric(df_ranks["WIDTH"], errors="coerce").fillna(0).astype(int)
    df_ranks["HEIGHT"] = pd.to_numeric(df_ranks["HEIGHT"], errors="coerce").fillna(0).astype(int)
    df_ranks["MAX_ITER"] = pd.to_numeric(df_ranks["MAX_ITER"], errors="coerce").fillna(0).astype(int)
    df_ranks["Rank"] = pd.to_numeric(df_ranks["Rank"], errors="coerce").fillna(0).astype(int)
    df_ranks["Code"] = df_ranks["Code"].astype(str)

    # Limpeza dos casos de resolução 512 (testes funcionais de corretude)
    if ignorar_512:
        df_exec = df_exec[df_exec["WIDTH"] > 512].copy()
        df_ranks = df_ranks[df_ranks["WIDTH"] > 512].copy()

    # Faz merge para herdar Scenario e Coordenadas em df_ranks
    cols_join = ["DataHora", "Code", "Schedule", "ChunkSize", "Threads", "WIDTH", "HEIGHT", "MAX_ITER"]
    df_ranks = pd.merge(df_ranks, df_exec[cols_join + ["Scenario", "RE_MIN", "RE_MAX", "Machine"]], on=cols_join, how="left")
    df_ranks["Scenario"] = df_ranks["Scenario"].fillna("Desconhecido")

    if machine and machine.lower() != "all":
        df_exec = df_exec[df_exec["Machine"] == machine].copy()
        df_ranks = df_ranks[df_ranks["Machine"] == machine].copy()

    return df_exec, df_ranks


def tratar_baselines_ponto_a_ponto(df_exec: pd.DataFrame) -> pd.DataFrame:
    """Extrai os tempos de referência T1 tratados para cada cenário e resolução via ponto_a_ponto_v2 (ou ponto_a_ponto)."""
    # Prioriza ponto_a_ponto_v2 como baseline mais recente e robusto, ou fallback para ponto_a_ponto
    if (df_exec["Code"] == "ponto_a_ponto_v2").any():
        base_code = "ponto_a_ponto_v2"
    else:
        base_code = "ponto_a_ponto"

    pap = df_exec[df_exec["Code"] == base_code].copy()
    cfg_cols = ["Scenario", "WIDTH", "HEIGHT", "MAX_ITER"]

    baselines = []
    for cfg_vals, group in pap.groupby(cfg_cols):
        tempos = group["TempoGasto"].values
        t_info = aplicar_tratativa(tempos)
        baselines.append({
            "Base_Code": base_code,
            "Scenario": cfg_vals[0],
            "WIDTH": cfg_vals[1],
            "HEIGHT": cfg_vals[2],
            "MAX_ITER": cfg_vals[3],
            "T1_Tratado": t_info["Tempo_Tratado"],
            "T1_Std": t_info["Std_Tratado"],
            "T1_CV_%": t_info["CV_Tratado_%"],
            "T1_Min_Desc": t_info["Min_Descartado"],
            "T1_Max_Desc": t_info["Max_Descartado"],
            "T1_Execucoes": t_info["N_Bruto"]
        })

    return pd.DataFrame(baselines)


def tratar_execucoes_mpi(df_exec: pd.DataFrame, df_baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica a tratativa aos tempos globais de execução de:
      - MPI (mpi_estatico, mpi_dinamico, mpi_dinamico_participativo, mpi_dinamico_meia_linha)
      - OpenMP (paralelo e paralelo_collapse)
      - ponto_a_ponto (Baseline T1)
    Calcula Speedup e Eficiência tratados em relação a T1.
    """
    df_parallel = df_exec[df_exec["Code"].isin([
        "mpi_estatico", "mpi_dinamico", "mpi_dinamico_participativo", "mpi_dinamico_meia_linha",
        "paralelo", "paralelo_collapse"
    ])].copy()
    cfg_cols = ["Code", "Scenario", "Schedule", "ChunkSize", "Threads", "WIDTH", "HEIGHT", "MAX_ITER"]

    records = []
    for cfg_vals, group in df_parallel.groupby(cfg_cols):
        tempos = group["TempoGasto"].values
        t_info = aplicar_tratativa(tempos)

        records.append({
            "Code": cfg_vals[0],
            "Scenario": cfg_vals[1],
            "Schedule": cfg_vals[2],
            "ChunkSize": cfg_vals[3],
            "Threads": cfg_vals[4],
            "WIDTH": cfg_vals[5],
            "HEIGHT": cfg_vals[6],
            "MAX_ITER": cfg_vals[7],
            "Tempo_Tratado": t_info["Tempo_Tratado"],
            "Std_Tratado": t_info["Std_Tratado"],
            "CV_Tratado_%": t_info["CV_Tratado_%"],
            "Min_Descartado": t_info["Min_Descartado"],
            "Max_Descartado": t_info["Max_Descartado"],
            "Execucoes_Brutas": t_info["N_Bruto"]
        })

    df_tratado = pd.DataFrame(records)

    # Mescla com baselines para calcular Speedup e Eficiência
    merge_cols = ["Scenario", "WIDTH", "HEIGHT", "MAX_ITER"]
    df_tratado = pd.merge(df_tratado, df_baselines[merge_cols + ["T1_Tratado"]], on=merge_cols, how="left")

    df_tratado["Speedup_Tratado"] = df_tratado["T1_Tratado"] / df_tratado["Tempo_Tratado"]
    df_tratado["Eficiencia_Tratada"] = df_tratado["Speedup_Tratado"] / df_tratado["Threads"]

    # Eficiência física: no caso de 4 núcleos/threads (HT em CPU de 2 núcleos físicos),
    # calcula tanto a eficiência por processo/thread lógico (Eficiencia_Tratada) quanto a física (Eficiencia_Fisica = Speedup / 2.0).
    def calc_eficiencia_fisica(r):
        threads = r["Threads"]
        sp = r["Speedup_Tratado"]
        if threads == 4:
            return sp / 2.0
        elif threads == 2:
            return sp / 2.0
        elif threads == 1:
            return 1.0
        return sp / min(threads, 2.0) if threads > 0 else np.nan

    df_tratado["Eficiencia_Fisica"] = df_tratado.apply(calc_eficiencia_fisica, axis=1)

    # Incorpora também os dados de baseline como linhas completas no dataset
    pap_records = []
    for _, row in df_baselines.iterrows():
        pap_records.append({
            "Code": row.get("Base_Code", "ponto_a_ponto"),
            "Scenario": row["Scenario"],
            "Schedule": "sequential",
            "ChunkSize": 0,
            "Threads": 1,
            "WIDTH": int(row["WIDTH"]),
            "HEIGHT": int(row["HEIGHT"]),
            "MAX_ITER": int(row["MAX_ITER"]),
            "Tempo_Tratado": row["T1_Tratado"],
            "Std_Tratado": row["T1_Std"],
            "CV_Tratado_%": row["T1_CV_%"],
            "Min_Descartado": row["T1_Min_Desc"],
            "Max_Descartado": row["T1_Max_Desc"],
            "Execucoes_Brutas": int(row["T1_Execucoes"]),
            "T1_Tratado": row["T1_Tratado"],
            "Speedup_Tratado": 1.0,
            "Eficiencia_Tratada": 1.0,
            "Eficiencia_Fisica": 1.0
        })

    df_pap = pd.DataFrame(pap_records)
    df_final = pd.concat([df_pap, df_tratado], ignore_index=True)

    # Ordenação inteligente: ponto_a_ponto primeiro por bloco de resolução, seguido de OpenMP e MPI
    def sort_order(code):
        if "ponto_a_ponto" in code:
            return 0
        elif "paralelo" in code:
            return 1
        return 2

    df_final["Paradigma_Ordem"] = df_final["Code"].apply(sort_order)
    df_final = df_final.sort_values(
        by=["Scenario", "WIDTH", "Paradigma_Ordem", "Tempo_Tratado"],
        ascending=[True, True, True, True]
    ).drop(columns=["Paradigma_Ordem"])

    return df_final


def tratar_tempos_por_rank(df_ranks: pd.DataFrame, df_baselines: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aplica a tratativa estatística aos tempos individuais de cada Rank/Thread/Processo
    e calcula métricas de balanceamento de carga entre os processos.
    """
    # 1. Tratativa por Rank individual de cada configuração
    cfg_rank_cols = ["Code", "Scenario", "Schedule", "ChunkSize", "Threads", "WIDTH", "HEIGHT", "MAX_ITER", "Rank"]
    rank_records = []

    for cfg_vals, group in df_ranks.groupby(cfg_rank_cols):
        tempos = group["TempoProcesso"].values
        t_info = aplicar_tratativa(tempos)

        rank_records.append({
            "Code": cfg_vals[0],
            "Scenario": cfg_vals[1],
            "Schedule": cfg_vals[2],
            "ChunkSize": cfg_vals[3],
            "Threads": cfg_vals[4],
            "WIDTH": cfg_vals[5],
            "HEIGHT": cfg_vals[6],
            "MAX_ITER": cfg_vals[7],
            "Rank": cfg_vals[8],
            "Tempo_Rank_Tratado": t_info["Tempo_Tratado"],
            "Std_Rank": t_info["Std_Tratado"],
            "CV_Rank_%": t_info["CV_Tratado_%"],
            "Min_Desc_Rank": t_info["Min_Descartado"],
            "Max_Desc_Rank": t_info["Max_Descartado"],
            "Repeticoes": t_info["N_Bruto"]
        })

    df_rank_tratado = pd.DataFrame(rank_records)

    # 2. Métricas de balanceamento de carga por configuração
    cfg_cols = ["Code", "Scenario", "Schedule", "ChunkSize", "Threads", "WIDTH", "HEIGHT", "MAX_ITER"]
    imbalance_records = []

    for cfg_vals, group in df_rank_tratado.groupby(cfg_cols):
        ranks_times = group.set_index("Rank")["Tempo_Rank_Tratado"].to_dict()
        all_times = list(ranks_times.values())
        workers_times = [t for r, t in ranks_times.items() if r > 0]

        t_coord = ranks_times.get(0, 0.0)
        t_max_rank = max(all_times) if all_times else 0.0
        t_min_rank = min(all_times) if all_times else 0.0
        t_mean_workers = float(np.mean(workers_times)) if workers_times else 0.0
        t_std_ranks = float(np.std(all_times, ddof=1)) if len(all_times) > 1 else 0.0
        t_std_workers = float(np.std(workers_times, ddof=1)) if len(workers_times) > 1 else 0.0

        # Taxa de desbalanceamento total (considerando todos os processos)
        imbalance_total_pct = ((t_max_rank - t_min_rank) / t_max_rank * 100.0) if t_max_rank > 0 else 0.0

        # Taxa de desbalanceamento estritamente entre trabalhadores (Ranks >= 1)
        if workers_times and max(workers_times) > 0:
            imbalance_workers_pct = ((max(workers_times) - min(workers_times)) / max(workers_times) * 100.0)
        else:
            imbalance_workers_pct = 0.0

        imbalance_records.append({
            "Code": cfg_vals[0],
            "Scenario": cfg_vals[1],
            "Schedule": cfg_vals[2],
            "ChunkSize": cfg_vals[3],
            "Threads": cfg_vals[4],
            "WIDTH": cfg_vals[5],
            "HEIGHT": cfg_vals[6],
            "MAX_ITER": cfg_vals[7],
            "Tempo_Coord_R0": t_coord,
            "Tempo_Max_Rank": t_max_rank,
            "Tempo_Min_Rank": t_min_rank,
            "Media_Trabalhadores": t_mean_workers,
            "Std_Entre_Ranks": t_std_ranks,
            "Std_Trabalhadores": t_std_workers,
            "Desbalanceamento_Total_%": imbalance_total_pct,
            "Desbalanceamento_Trabalhadores_%": imbalance_workers_pct
        })

    df_imbalance = pd.DataFrame(imbalance_records)

    # Combina para ter a visão consolidada
    merged = pd.merge(df_rank_tratado, df_imbalance, on=cfg_cols, how="left")

    # Incorpora também ponto_a_ponto como Rank 0 referencial caso fornecido
    if df_baselines is not None and not df_baselines.empty:
        pap_ranks = []
        for _, b in df_baselines.iterrows():
            pap_ranks.append({
                "Code": b.get("Base_Code", "ponto_a_ponto"),
                "Scenario": b["Scenario"],
                "Schedule": "sequential",
                "ChunkSize": 0,
                "Threads": 1,
                "WIDTH": int(b["WIDTH"]),
                "HEIGHT": int(b["HEIGHT"]),
                "MAX_ITER": int(b["MAX_ITER"]),
                "Rank": 0,
                "Tempo_Rank_Tratado": b["T1_Tratado"],
                "Std_Rank": b["T1_Std"],
                "CV_Rank_%": b["T1_CV_%"],
                "Min_Desc_Rank": b["T1_Min_Desc"],
                "Max_Desc_Rank": b["T1_Max_Desc"],
                "Repeticoes": int(b["T1_Execucoes"]),
                "Tempo_Coord_R0": b["T1_Tratado"],
                "Tempo_Max_Rank": b["T1_Tratado"],
                "Tempo_Min_Rank": b["T1_Tratado"],
                "Media_Trabalhadores": 0.0,
                "Std_Entre_Ranks": 0.0,
                "Std_Trabalhadores": 0.0,
                "Desbalanceamento_Total_%": 0.0,
                "Desbalanceamento_Trabalhadores_%": 0.0
            })
        merged = pd.concat([pd.DataFrame(pap_ranks), merged], ignore_index=True)

    merged["Is_Baseline"] = (merged["Code"].str.contains("ponto_a_ponto")).astype(int)
    merged = merged.sort_values(
        by=["Scenario", "WIDTH", "Is_Baseline", "Code", "Schedule", "ChunkSize", "Rank"],
        ascending=[True, True, False, True, True, True, True]
    ).drop(columns=["Is_Baseline"])

    return merged


def print_separator(title: str = "", char: str = "=", width: int = 105):
    if title:
        print("\n" + char * width)
        print(f" {title.upper()}")
        print(char * width)
    else:
        print(char * width)


def exibir_relatorio(df_baselines: pd.DataFrame, df_mpi: pd.DataFrame, df_ranks: pd.DataFrame, ignorou_512: bool = True):
    """Exibe um relatório estruturado no terminal com tabelas claras."""
    print_separator("📊 RELATÓRIO DE TRATAMENTO ESTATÍSTICO DE DADOS (PONTO_A_PONTO E MPI)", char="=")
    print(" • Método Adotado  : Média dos 3 tempos centrais (descarte sistemático de t_min e t_max)")
    print(" • Métricas Finais : Tempo Tratado (s), Desvio Padrão (s), CV (%), Speedup e Eficiência")
    print(" • Escopo          : ponto_a_ponto (Baseline) + MPI (Tempo Total e Tempos por Processo/Thread)")
    if ignorou_512:
        print(" • Filtro Aplicado : 🧹 Casos de resolução 512x512 removidos (foco exclusivo em benchmarks 4K, 8K e 16K)")
    else:
        print(" • Filtro Aplicado : Nenhum (incluindo testes de resolução 512x512)")

    # 1. Baselines Tratados
    print_separator("1. Tempos de Referência Tratados (ponto_a_ponto - T1 Baseline)", char="-")
    cols_base = ["Scenario", "WIDTH", "T1_Tratado", "T1_Std", "T1_CV_%", "T1_Min_Desc", "T1_Max_Desc", "T1_Execucoes"]
    print(df_baselines[cols_base].to_string(index=False))

    # 2. Resumo dos Melhores Algoritmos MPI por Resolução e Cenário
    print_separator("2. Melhores Configurações MPI Tratadas por Resolução e Cenário (4 Processos)", char="-")
    df_mpi_only = df_mpi[df_mpi["Code"].str.startswith("mpi_") & (df_mpi["Threads"] == 4)].copy()
    if not df_mpi_only.empty:
        idx_best_mpi = df_mpi_only.groupby(["Scenario", "WIDTH"])["Tempo_Tratado"].idxmin()
        best_mpi = df_mpi_only.loc[idx_best_mpi].copy()
        cols_best = ["Scenario", "WIDTH", "Code", "Schedule", "ChunkSize", "Tempo_Tratado", "Std_Tratado", "CV_Tratado_%", "Speedup_Tratado", "Eficiencia_Tratada"]
        print(best_mpi[cols_best].to_string(index=False))

    # 3. Resumo dos Melhores Algoritmos OpenMP por Resolução e Cenário
    print_separator("3. Melhores Configurações OpenMP Tratadas por Resolução e Cenário (4 Threads)", char="-")
    df_omp_only = df_mpi[df_mpi["Code"].isin(["paralelo", "paralelo_collapse"]) & (df_mpi["Threads"] == 4)].copy()
    if not df_omp_only.empty:
        idx_best_omp = df_omp_only.groupby(["Scenario", "WIDTH"])["Tempo_Tratado"].idxmin()
        best_omp = df_omp_only.loc[idx_best_omp].copy()
        cols_best_omp = ["Scenario", "WIDTH", "Code", "Schedule", "ChunkSize", "Tempo_Tratado", "Std_Tratado", "CV_Tratado_%", "Speedup_Tratado", "Eficiencia_Tratada"]
        print(best_omp[cols_best_omp].to_string(index=False))

    # 4. Comparativo Direto: OpenMP vs MPI (4 Threads/Processos)
    if not df_mpi_only.empty and not df_omp_only.empty:
        print_separator("4. Comparativo Direto de Paradigma: OpenMP vs MPI (4 Núcleos)", char="-")
        comp = pd.merge(
            best_omp[["Scenario", "WIDTH", "Code", "Schedule", "ChunkSize", "Tempo_Tratado", "Speedup_Tratado", "Eficiencia_Tratada"]],
            best_mpi[["Scenario", "WIDTH", "Code", "Schedule", "ChunkSize", "Tempo_Tratado", "Speedup_Tratado", "Eficiencia_Tratada"]],
            on=["Scenario", "WIDTH"],
            suffixes=("_OMP", "_MPI")
        )
        comp["Diferenca_s"] = comp["Tempo_Tratado_MPI"] - comp["Tempo_Tratado_OMP"]
        comp["Vantagem_OMP_%"] = (comp["Tempo_Tratado_MPI"] - comp["Tempo_Tratado_OMP"]) / comp["Tempo_Tratado_MPI"] * 100.0

        for _, r in comp.iterrows():
            print(f" • [{r['Scenario']} {r['WIDTH']}x{r['WIDTH']}]:")
            print(f"     - OpenMP : {r['Tempo_Tratado_OMP']:>8.4f}s | Speedup: {r['Speedup_Tratado_OMP']:>5.2f}x (Efic: {r['Eficiencia_Tratada_OMP']*100:>5.1f}%) | Config: {r['Code_OMP']} {r['Schedule_OMP']}({r['ChunkSize_OMP']})")
            print(f"     - MPI    : {r['Tempo_Tratado_MPI']:>8.4f}s | Speedup: {r['Speedup_Tratado_MPI']:>5.2f}x (Efic: {r['Eficiencia_Tratada_MPI']*100:>5.1f}%) | Config: {r['Code_MPI']} {r['Schedule_MPI']}({r['ChunkSize_MPI']})")
            print(f"     - Ganho  : OpenMP foi {r['Diferenca_s']:>6.3f}s mais rápido ({r['Vantagem_OMP_%']:>5.1f}% de vantagem sobre o MPI)")

    # 5. Desempenho Geral Médio por Código (Todos os Paradigmas)
    print_separator("5. Desempenho Geral Médio por Código (Dados Tratados)", char="-")
    resumo_alg = df_mpi.groupby("Code").agg(
        Configs=("WIDTH", "count"),
        Tempo_Medio=("Tempo_Tratado", "mean"),
        Std_Medio=("Std_Tratado", "mean"),
        CV_Medio=("CV_Tratado_%", "mean"),
        Speedup_Medio=("Speedup_Tratado", "mean"),
        Eficiencia_Media=("Eficiencia_Tratada", "mean")
    ).reset_index()
    print(resumo_alg.to_string(index=False))

    # 6. Análise de Balanceamento de Carga entre Processos (Ranks)
    print_separator("6. Análise de Desbalanceamento de Carga entre Ranks (Tempos por Processo Tratados)", char="-")
    print("Média do desbalanceamento (Workers Imbalance %) e dispersão entre trabalhadores por algoritmo:")
    resumo_imb = df_ranks.drop_duplicates(subset=["Code", "Scenario", "Schedule", "ChunkSize", "WIDTH"]).groupby("Code").agg(
        Casos=("WIDTH", "count"),
        Tempo_Coord_Medio=("Tempo_Coord_R0", "mean"),
        Tempo_Max_Processo=("Tempo_Max_Rank", "mean"),
        Media_Trabalhadores=("Media_Trabalhadores", "mean"),
        Desbalanceamento_Workers_Pct=("Desbalanceamento_Trabalhadores_%", "mean"),
        Std_Entre_Workers=("Std_Trabalhadores", "mean")
    ).reset_index()
    print(resumo_imb.to_string(index=False))

    # 7. Casos Práticos: Estático (Block vs Cyclic) e Dinâmico
    print_separator("7. Amostras Práticas de Balanceamento por Rank (Resolução 8192x8192 - Zoom Seahorse)", char="-")
    sub_ex = df_ranks[(df_ranks["WIDTH"] == 8192) & (df_ranks["Scenario"] == "Zoom (Seahorse)")]
    exemplos_codigos = ["mpi_estatico", "mpi_dinamico", "mpi_dinamico_participativo", "mpi_dinamico_meia_linha"]

    for cod in exemplos_codigos:
        grupo_cod = sub_ex[sub_ex["Code"] == cod]
        if grupo_cod.empty:
            continue
        scheds = grupo_cod["Schedule"].unique()
        for sched in scheds:
            sub_sched = grupo_cod[grupo_cod["Schedule"] == sched]
            primeiro_cfg = sub_sched.iloc[0]
            chunk = primeiro_cfg["ChunkSize"]
            ranks_cfg = sub_sched[(sub_sched["ChunkSize"] == chunk)].drop_duplicates(subset=["Rank"]).sort_values(by="Rank")

            tempos_str = ", ".join([f"R{int(r['Rank'])}: {r['Tempo_Rank_Tratado']:.3f}s" for _, r in ranks_cfg.iterrows()])
            print(f" • [{cod:<26}] {sched:<12} (ch={chunk:>2}): {tempos_str} | Desbal. Workers: {primeiro_cfg['Desbalanceamento_Trabalhadores_%']:.2f}%")

    print_separator("✅ TRATATIVA CONCLUÍDA COM SUCESSO", char="=")


def exportar_arquivos(df_mpi: pd.DataFrame, df_ranks: pd.DataFrame, out_dir: str = "data"):
    """Exporta os dados tratados para CSVs limpos."""
    os.makedirs(out_dir, exist_ok=True)
    path_mpi = os.path.join(out_dir, "dados_tratados_mpi.csv")
    path_ranks = os.path.join(out_dir, "dados_tratados_ranks_mpi.csv")

    df_mpi.to_csv(path_mpi, index=False)
    df_ranks.to_csv(path_ranks, index=False)

    print(f"\n📁 Arquivos tratados exportados:")
    print(f"   1. {path_mpi} ({len(df_mpi)} configurações tratadas)")
    print(f"   2. {path_ranks} ({len(df_ranks)} registros por rank tratados)")


def main():
    parser = argparse.ArgumentParser(description="Tratamento Estatístico de Dados de Benchmark (Mandelbrot MPI e ponto_a_ponto)")
    parser.add_argument("--exec-csv", default="data/dateTimeExecution.csv", help="Caminho do CSV de execuções globais")
    parser.add_argument("--ranks-csv", default="data/tempos_processos_mpi.csv", help="Caminho do CSV de tempos por processo")
    parser.add_argument("--machine", default="all", help="Filtrar por máquina (padrão: all, ex: deCasa)")
    parser.add_argument("--incluir-512", action="store_true", help="Incluir casos de teste de resolução 512 (por padrão são removidos)")
    parser.add_argument("--out-dir", default="data", help="Diretório de saída para os CSVs tratados (padrão: data)")
    parser.add_argument("--no-export", action="store_true", help="Não exportar arquivos CSV, apenas exibir relatório no terminal")

    args = parser.parse_args()
    ignorar_512 = not args.incluir_512

    # 1. Carrega dados brutos (com limpeza automática de 512)
    df_exec, df_ranks = carregar_dados(
        exec_path=args.exec_csv,
        ranks_path=args.ranks_csv,
        machine=args.machine,
        ignorar_512=ignorar_512
    )

    # 2. Tratativa dos Baselines (ponto_a_ponto)
    df_baselines = tratar_baselines_ponto_a_ponto(df_exec)

    # 3. Tratativa dos Tempos Globais MPI (Speedup e Eficiência)
    df_mpi_tratado = tratar_execucoes_mpi(df_exec, df_baselines)

    # 4. Tratativa dos Tempos por Processo/Rank/Thread e Balanceamento
    df_ranks_tratado = tratar_tempos_por_rank(df_ranks, df_baselines=df_baselines)

    # 5. Exibe relatório formatado
    exibir_relatorio(df_baselines, df_mpi_tratado, df_ranks_tratado, ignorou_512=ignorar_512)

    # 6. Exporta arquivos CSV
    if not args.no_export:
        exportar_arquivos(df_mpi_tratado, df_ranks_tratado, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
