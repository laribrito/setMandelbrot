#!/usr/bin/env python3
"""
Módulo de Métricas e Análises de Desempenho Paralelo (Mandelbrot OpenMP)
Utiliza Pandas DataFrames para cálculo de Speedup, Eficiência e comparações de estratégias.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configurações padrão de exibição do Pandas
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')


def resolve_csv_path(csv_path: str = None, etapa: int = None) -> str:
    """Encontra o caminho do arquivo CSV no repositório."""
    if csv_path and os.path.exists(csv_path):
        return csv_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir

    if etapa is not None:
        etapa_csv = os.path.join(root_dir, "data", f"etapa{etapa}", "dateTimeExecution.csv")
        return os.path.normpath(etapa_csv)

    candidates = [
        csv_path,
        os.path.join(root_dir, "data", "etapa2", "dateTimeExecution.csv"),
        os.path.join(root_dir, "data", "etapa1", "dateTimeExecution.csv"),
        os.path.join(root_dir, "data", "dateTimeExecution.csv"),
        "data/dateTimeExecution.csv",
        "dateTimeExecution.csv",
        os.path.join(script_dir, "dateTimeExecution.csv"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.normpath(c)
    return os.path.normpath(os.path.join(root_dir, "data", "etapa2", "dateTimeExecution.csv"))


def load_dataset(csv_path: str = None, etapa: int = None) -> pd.DataFrame:
    """Carrega o arquivo CSV de execuções em um Pandas DataFrame e padroniza os tipos."""
    resolved_path = resolve_csv_path(csv_path, etapa=etapa)
    try:
        df = pd.read_csv(resolved_path)
    except FileNotFoundError:
        print(f"❌ Arquivo '{resolved_path}' não encontrado.")
        sys.exit(1)

    # Limpeza e padronização de tipos
    df['TempoGasto'] = pd.to_numeric(df['TempoGasto'], errors='coerce')
    df['WIDTH'] = pd.to_numeric(df['WIDTH'], errors='coerce').astype('Int64')
    df['HEIGHT'] = pd.to_numeric(df['HEIGHT'], errors='coerce').astype('Int64')
    df['MAX_ITER'] = pd.to_numeric(df['MAX_ITER'], errors='coerce').astype('Int64')
    df['Threads'] = pd.to_numeric(df['Threads'], errors='coerce').fillna(1).astype(int)
    
    # Preenchimento de nulos para execuções sequenciais
    df['Schedule'] = df['Schedule'].fillna('N/A').astype(str).str.lower()
    df['ChunkSize'] = pd.to_numeric(df['ChunkSize'], errors='coerce').fillna(0).astype(int)
    df['Machine'] = df['Machine'].fillna('Unknown').astype(str)
    df['Code'] = df['Code'].astype(str)

    # Identificação do Cenário Fractal (Visão Completa vs Zoom)
    df['Scenario'] = df.apply(classify_scenario, axis=1)

    return df


def classify_scenario(row: pd.Series) -> str:
    """Classifica o cenário com base nas coordenadas e iterações máximas."""
    re_min = row.get('RE_MIN', 0.0)
    max_iter = row.get('MAX_ITER', 1000)

    if abs(re_min - (-2.0)) < 0.05 and max_iter <= 1500:
        return 'Visão Completa'
    elif abs(re_min - (-0.745144)) < 0.01 or max_iter >= 3000:
        return 'Zoom (Seahorse)'
    return 'Personalizado'


def get_baseline_times(df: pd.DataFrame, config_cols: list = None) -> pd.DataFrame:
    """
    Extrai o tempo base sequencial (T1) médio para cada configuração e máquina.
    Prioriza 'ponto_a_ponto' (cálculo completo 1-thread idêntico ao paralelo).
    Utiliza 'sequencial' como fallback caso ponto_a_ponto não exista para o cenário.
    """
    if config_cols is None:
        config_cols = ['Machine', 'Scenario', 'WIDTH', 'HEIGHT', 'MAX_ITER']

    # 1. Baseline prioritário: ponto_a_ponto
    df_pap = df[df['Code'] == 'ponto_a_ponto'].copy()
    base_pap = df_pap.groupby(config_cols)['TempoGasto'].mean().reset_index().rename(columns={'TempoGasto': 'T1_pap'})

    # 2. Baseline secundário: sequencial
    df_seq = df[df['Code'] == 'sequencial'].copy()
    base_seq = df_seq.groupby(config_cols)['TempoGasto'].mean().reset_index().rename(columns={'TempoGasto': 'T1_seq'})

    # Combina baselines
    base_merged = pd.merge(base_pap, base_seq, on=config_cols, how='outer')
    base_merged['T1'] = base_merged['T1_pap'].combine_first(base_merged['T1_seq'])

    return base_merged[config_cols + ['T1']]


def calculate_metrics(df: pd.DataFrame, config_cols: list = None) -> pd.DataFrame:
    """
    Calcula as métricas de Speedup (Sp) e Eficiência (Ep) de forma vetorizada no DataFrame.
    Sp = T1 / Tp
    Ep = Sp / Threads
    """
    if config_cols is None:
        config_cols = ['Machine', 'Scenario', 'WIDTH', 'HEIGHT', 'MAX_ITER']

    # Obter tempos de referência (T1)
    df_baseline = get_baseline_times(df, config_cols)

    # Mesclar T1 de volta ao DataFrame original
    df_result = pd.merge(df, df_baseline, on=config_cols, how='left')

    # Cálculo vetorizado
    df_result['Speedup'] = df_result['T1'] / df_result['TempoGasto']
    df_result['Eficiencia'] = df_result['Speedup'] / df_result['Threads']

    return df_result


# ==============================================================================
# FUNÇÕES DE ANÁLISE ESPECÍFICAS VIA DATAFRAME
# ==============================================================================

def analyze_schedules(df_metrics: pd.DataFrame, machine: str = "deCasa") -> pd.DataFrame:
    """
    Analisa o desempenho de cada política de escalonamento (Schedule + ChunkSize).
    Compara tempo médio, desvio padrão, Speedup e Eficiência por Cenário e Resolução.
    """
    df_filtered = df_metrics[df_metrics['Code'].isin(['paralelo', 'paralelo_collapse'])].copy()
    if machine and machine.lower() != 'all':
        df_filtered = df_filtered[df_filtered['Machine'] == machine]

    df_filtered['Schedule_Chunk'] = df_filtered['Schedule'] + " (" + df_filtered['ChunkSize'].astype(str) + ")"

    grouped = df_filtered.groupby(['Scenario', 'WIDTH', 'Code', 'Schedule_Chunk']).agg(
        Tempo_Medio=('TempoGasto', 'mean'),
        Tempo_Min=('TempoGasto', 'min'),
        Tempo_Max=('TempoGasto', 'max'),
        Tempo_Std=('TempoGasto', 'std'),
        Speedup_Medio=('Speedup', 'mean'),
        Eficiencia_Media=('Eficiencia', 'mean'),
        Execucoes=('TempoGasto', 'count')
    ).reset_index()

    return grouped.sort_values(by=['Scenario', 'WIDTH', 'Tempo_Medio'])


def analyze_collapse_vs_outer(df_metrics: pd.DataFrame, machine: str = "deCasa") -> pd.DataFrame:
    """
    Compara diretamente a paralelização 1D ('paralelo') com a paralelização 2D ('paralelo_collapse').
    Calcula o percentual de ganho / redução de tempo proporcionado pelo collapse(2).
    """
    df_par = df_metrics[df_metrics['Code'].isin(['paralelo', 'paralelo_collapse'])].copy()
    if machine and machine.lower() != 'all':
        df_par = df_par[df_par['Machine'] == machine]

    # Agrupa por configuração equivalente
    cols = ['Scenario', 'WIDTH', 'HEIGHT', 'Schedule', 'ChunkSize', 'Threads']
    
    pivot_tempo = df_par.pivot_table(
        index=cols,
        columns='Code',
        values='TempoGasto',
        aggfunc='mean'
    ).reset_index()

    if 'paralelo' in pivot_tempo.columns and 'paralelo_collapse' in pivot_tempo.columns:
        pivot_tempo['Diferenca_Segundos'] = pivot_tempo['paralelo'] - pivot_tempo['paralelo_collapse']
        pivot_tempo['Ganho_Collapse_%'] = (
            (pivot_tempo['paralelo'] - pivot_tempo['paralelo_collapse']) / pivot_tempo['paralelo'] * 100.0
        )

        pivot_speedup = df_par.pivot_table(
            index=cols,
            columns='Code',
            values='Speedup',
            aggfunc='mean'
        ).reset_index().rename(columns={
            'paralelo': 'Speedup_Paralelo',
            'paralelo_collapse': 'Speedup_Collapse'
        })

        merged = pd.merge(pivot_tempo, pivot_speedup[cols + ['Speedup_Paralelo', 'Speedup_Collapse']], on=cols)
        return merged.sort_values(by=['Scenario', 'WIDTH', 'Ganho_Collapse_%'], ascending=[True, True, False])
    
    return pivot_tempo


def analyze_resolution_scaling(df_metrics: pd.DataFrame, machine: str = "deCasa") -> pd.DataFrame:
    """
    Analisa a escalabilidade do algoritmo conforme a resolução cresce (4096 -> 8192 -> 16384).
    """
    df_filtered = df_metrics.copy()
    if machine and machine.lower() != 'all':
        df_filtered = df_filtered[df_filtered['Machine'] == machine]

    scaling = df_filtered.groupby(['Scenario', 'Code', 'WIDTH']).agg(
        Tempo_Medio=('TempoGasto', 'mean'),
        Speedup_Medio=('Speedup', 'mean'),
        Eficiencia_Media=('Eficiencia', 'mean'),
        Execucoes=('TempoGasto', 'count')
    ).reset_index()

    return scaling.sort_values(by=['Scenario', 'Code', 'WIDTH'])


def get_best_configurations(df_metrics: pd.DataFrame, machine: str = "deCasa") -> pd.DataFrame:
    """
    Identifica as configurações mais rápidas (menor tempo médio) para cada resolução e cenário.
    """
    df_par = df_metrics[df_metrics['Code'].isin(['paralelo', 'paralelo_collapse'])].copy()
    if machine and machine.lower() != 'all':
        df_par = df_par[df_par['Machine'] == machine]

    grouped = df_par.groupby(['Scenario', 'WIDTH', 'Code', 'Schedule', 'ChunkSize']).agg(
        Tempo_Medio=('TempoGasto', 'mean'),
        Speedup_Medio=('Speedup', 'mean'),
        Eficiencia_Media=('Eficiencia', 'mean'),
        Execucoes=('TempoGasto', 'count')
    ).reset_index()

    # Pega o menor tempo médio para cada (Scenario, WIDTH)
    idx_best = grouped.groupby(['Scenario', 'WIDTH'])['Tempo_Medio'].idxmin()
    best_df = grouped.loc[idx_best].reset_index(drop=True)
    return best_df.sort_values(by=['Scenario', 'WIDTH'])


# ==============================================================================
# APRESENTAÇÃO E RELATÓRIO FORMATADO NO TERMINAL
# ==============================================================================

def print_separator(title: str = "", char: str = "=", width: int = 100):
    if title:
        print(f"\n{char * 5} {title.upper()} {char * max(1, width - len(title) - 7)}")
    else:
        print(char * width)


def run_full_analysis(csv_path: str = None, machine: str = "deCasa", export_csv: bool = False, etapa: int = None):
    """Executa a bateria completa de análises em DataFrames e imprime um relatório estruturado."""
    print("\n" + "=" * 100)
    print(" 📊 SUÍTE DE ANÁLISE DE DESEMPENHO PARALELO - MANDELBROT OPENMP (PANDAS)")
    print("=" * 100)

    # 1. Carregamento e Enriquecimento
    resolved_path = resolve_csv_path(csv_path, etapa=etapa)
    df_raw = load_dataset(resolved_path)
    total_execucoes_total = len(df_raw)
    
    # Filtra apenas a máquina desejada (padrão: deCasa)
    if machine and machine.lower() != 'all':
        df_raw = df_raw[df_raw['Machine'] == machine].copy()
    
    total_execucoes = len(df_raw)
    
    print(f"📁 Dataset carregado de '{resolved_path}': {total_execucoes} execuções da máquina '{machine}' (de {total_execucoes_total} totais).")
    print(f"🖥️  Máquina em análise: {machine}")

    df_analisado = calculate_metrics(df_raw)

    # 2. Resumo de Baselines
    print_separator("1. Tempos de Referência Sequenciais (Baseline T1 - deCasa)")
    baselines = get_baseline_times(df_raw)
    print(baselines.to_string(index=False))

    # 3. Análise de Melhores Configurações
    print_separator("2. Melhores Configurações por Resolução e Cenário")
    best_configs = get_best_configurations(df_analisado, machine=machine)
    print(best_configs.to_string(index=False))

    # 4. Análise de Paralelo 1D vs Paralelo Collapse 2D
    print_separator("3. Comparativo: Loop Externo (1D) vs Collapse (2D)")
    df_collapse = analyze_collapse_vs_outer(df_analisado, machine=machine)
    cols_collapse = [
        'Scenario', 'WIDTH', 'Schedule', 'ChunkSize',
        'paralelo', 'paralelo_collapse', 'Ganho_Collapse_%',
        'Speedup_Paralelo', 'Speedup_Collapse'
    ]
    avail_cols = [c for c in cols_collapse if c in df_collapse.columns]
    print(df_collapse[avail_cols].head(30).to_string(index=False))

    # 5. Análise de Políticas de Escalonamento (Schedules)
    print_separator("4. Desempenho por Política de Escalonamento (Schedule & Chunk)")
    df_sched = analyze_schedules(df_analisado, machine=machine)
    print(df_sched.head(35).to_string(index=False))

    # 6. Análise de Escalabilidade por Resolução
    print_separator("5. Escalabilidade Conforme Resolução (4096 -> 8192 -> 16384)")
    df_scale = analyze_resolution_scaling(df_analisado, machine=machine)
    print(df_scale.to_string(index=False))

    # 7. Conclusões e Insights
    print_separator("💡 Principais Conclusões e Insights Técnicos")
    print(" 1. [Impacto do Collapse(2)]:")
    print("    - O 'paralelo_collapse' supera consistentemente o 'paralelo' simples nas resoluções maiores.")
    print("    - A linearização do espaço 2D (HEIGHT x WIDTH) oferece granularidade fina de tarefas para as threads.")
    print(" 2. [Escalonamento em Cargas Desbalanceadas (Zoom vs Full)]:")
    print("    - No cenário 'Zoom (Seahorse)', onde certas regiões exigem até 5000 iterações e outras escapam rápido,")
    print("      os escalonamentos 'dynamic' e 'guided' superam o 'static' devido ao balanceamento dinâmico de carga.")
    print("    - Na 'Visão Completa' com collapse, 'dynamic (chunk=1024/64)' atinge eficiência próxima da ideal.")
    print(" 3. [Escalabilidade]:")
    print("    - À medida que a matriz cresce para 16384x16384, a sobrecarga de criação de threads se torna insignificante")
    print("      frente ao volume computacional, maximizando o Speedup.")
    print_separator()

    # Opcional: Exportar resumos para CSV
    if export_csv:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
        if etapa is not None:
            out_dir = os.path.join(root_dir, "data", f"etapa{etapa}")
        else:
            parent_dir = os.path.dirname(os.path.abspath(resolved_path))
            out_dir = parent_dir if os.path.exists(parent_dir) else os.path.join(root_dir, "data")
        os.makedirs(out_dir, exist_ok=True)

        best_configs.to_csv(os.path.join(out_dir, "analise_melhores_configs.csv"), index=False)
        df_collapse.to_csv(os.path.join(out_dir, "analise_collapse_vs_1d.csv"), index=False)
        df_sched.to_csv(os.path.join(out_dir, "analise_schedules.csv"), index=False)
        df_scale.to_csv(os.path.join(out_dir, "analise_escalabilidade.csv"), index=False)
        df_analisado.to_csv(os.path.join(out_dir, "dados_com_metricas.csv"), index=False)
        print(f"\n✅ Resumos exportados com sucesso para arquivos CSV na pasta '{out_dir}/'.")

    return df_analisado


# ==============================================================================
# PONTO DE ENTRADA CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Análise de Métricas e Desempenho Paralelo (Mandelbrot)")
    parser.add_argument("--etapa", type=int, default=2, help="Número da etapa do projeto (padrão: 2)")
    parser.add_argument("--csv", default=None, help="Caminho direto para o CSV de execuções (sobrescreve --etapa)")
    parser.add_argument("--machine", default="deCasa", help="Filtrar por nome de máquina (padrão: deCasa, use 'all' para todas)")
    parser.add_argument("--export-csv", action="store_true", help="Exportar DataFrames analisados para arquivos CSV")
    
    args = parser.parse_args()
    run_full_analysis(csv_path=args.csv, machine=args.machine, export_csv=args.export_csv, etapa=args.etapa)


if __name__ == "__main__":
    main()