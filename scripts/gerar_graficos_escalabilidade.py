#!/usr/bin/env python3
"""
Geração de Gráficos e Análise de Escalabilidade: Leis de Amdahl e Gustafson (MPI Mandelbrot)
-----------------------------------------------------------------------------------------
Este script realiza a análise científica rigorosa de desempenho paralelo em dois regimes:

1. ESCALABILIDADE FORTE (Strong Scaling - Lei de Amdahl):
   - O tamanho global do problema permanece estritamente FIXO (4K, 8K, 16K).
   - Varia-se a quantidade de unidades de processamento (p).
   - Modela a fração estritamente serial (s) e a fração paralelizável (f = 1 - s)
     via métrica de Karp-Flatt: s = (1/Sp - 1/p) / (1 - 1/p).
   - Plota Speedup e Eficiência reais confrontados com a curva teórica de Amdahl
     S(p) = 1 / (s + (1-s)/p) e seu limite assintótico teórico S_max = 1/s.

2. ESCALABILIDADE FRACA (Weak Scaling - Lei de Gustafson-Barsis):
   - A carga total de trabalho se expande PROPORCIONALMENTE à infraestrutura (p).
   - A carga por unidade de processamento permanece estritamente constante.
   - No Mandelbrot 2D (pixels = W x H), a expansão de p=1 para p=4 corresponde
     exatamente ao salto de 4K (4096x4096 = 16.78M px) para 8K (8192x8192 = 67.11M px),
     mantendo 16.78M pixels por processo.
   - Calcula a Eficiência de Escalabilidade Fraca: E_weak = T1(W1) / Tp(p * W1).
   - Calcula o Speedup Escalado de Gustafson: S_scaled = p * E_weak.
   - Compara com a Lei de Gustafson Teórica: S_Gustafson(p) = s + p*(1 - s) = p - s*(p - 1).

Gera:
   - relatorios/png/ e relatorios/pdf/:
     * mpi_escalabilidade_forte_amdahl.{png,pdf}
     * mpi_escalabilidade_fraca_gustafson.{png,pdf}
     * mpi_comparativo_amdahl_gustafson.{png,pdf}
     * mpi_escalabilidade_resolucoes_4k_8k_16k.{png,pdf}
     * mpi_escalabilidade_forte_512_p1_a_p4.{png,pdf}
   - data/analise_amdal_gustafson.csv (resumo numérico consolidado)
"""

import os
import sys

# Configura diretório de cache do matplotlib para evitar avisos em ambientes restritos
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==============================================================================
# CONFIGURAÇÃO VISUAL MODERNA E PALETA INSTITUCIONAL
# ==============================================================================
def configurar_estilo():
    """Configura estética moderna, tipografia limpa e grids suaves."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
    plt.rcParams['axes.edgecolor'] = '#94a3b8'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.color'] = '#e2e8f0'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.75
    plt.rcParams['figure.autolayout'] = False


CORES_ESTRATEGIAS = {
    'Estático Bloco': '#64748b',       # Cinza ardósia
    'Estático Cíclico': '#2563eb',      # Azul real
    'Dinâmico Dedicado': '#059669',     # Verde esmeralda
    'Dinâmico Meia-Linha': '#d97706',   # Âmbar / Laranja
    'Dinâmico Participativo': '#dc2626',# Vermelho carmesim
    'Ideal Linear': '#ef4444',          # Vermelho vibrante
    'Amdahl Teórico': '#8b5cf6',        # Violeta
    'Gustafson Teórico': '#06b6d4',     # Ciano
    'Referência Serial (p=1)': '#0f172a'# Grafite escuro
}

MARCADORES = {
    'Estático Bloco': 's',
    'Estático Cíclico': 'o',
    'Dinâmico Dedicado': '^',
    'Dinâmico Meia-Linha': 'D',
    'Dinâmico Participativo': 'v',
    'Referência Serial (p=1)': 'P'
}


# ==============================================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================
def resolve_path(rel_path: str) -> str:
    """Resolve caminho relativo a partir da raiz do repositório."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    return os.path.normpath(os.path.join(root_dir, rel_path))


def carregar_dados():
    """Carrega dados consolidados de tabela_geral_mpi.csv e dateTimeExecution.csv."""
    tabela_path = resolve_path("data/tabela_geral_mpi.csv")
    exec_path = resolve_path("data/dateTimeExecution.csv")

    if not os.path.exists(tabela_path):
        raise FileNotFoundError(f"Arquivo não encontrado: '{tabela_path}'. Execute 'gerar_tabelas_csv.py' primeiro.")

    df_geral = pd.read_csv(tabela_path)

    # Identifica o melhor chunk de cada estratégia em cada (Cenário, Resolução)
    mpi_only = df_geral[df_geral["Code"] != "ponto_a_ponto_v2"].copy()
    idx_best = mpi_only.groupby(["Cenário", "Estratégia", "Resolução"])["Speedup"].idxmax()
    df_best = mpi_only.loc[idx_best].copy()

    # Mapeamento do T1 canônico por cenário e resolução
    t1_records = df_geral[df_geral["Code"] == "ponto_a_ponto_v2"][["Cenário", "Resolução", "Tempo_s"]].drop_duplicates()
    t1_map = {}
    for _, row in t1_records.iterrows():
        # Ex: ('Visão Completa', 4096)
        res_num = int(row["Resolução"].split("×")[0].strip())
        t1_map[(row["Cenário"], res_num)] = float(row["Tempo_s"])

    # Carrega dados brutos para 512x512 com variação empírica de processos p=1,2,3,4
    df_512 = pd.DataFrame()
    if os.path.exists(exec_path):
        raw_df = pd.read_csv(exec_path)
        raw_df["WIDTH"] = pd.to_numeric(raw_df["WIDTH"], errors="coerce")
        raw_df["Threads"] = pd.to_numeric(raw_df["Threads"], errors="coerce").fillna(1).astype(int)
        raw_df["TempoGasto"] = pd.to_numeric(raw_df["TempoGasto"], errors="coerce")

        def classify(r):
            re_min = r.get("RE_MIN", 0.0)
            max_it = r.get("MAX_ITER", 1000)
            if abs(re_min - (-2.0)) < 0.05 and max_it <= 1500:
                return "Visão Completa"
            elif abs(re_min - (-0.745144)) < 0.01 or max_it >= 3000:
                return "Zoom (Seahorse)"
            return "Outro"

        raw_df["Cenário"] = raw_df.apply(classify, axis=1)
        df_512 = raw_df[raw_df["WIDTH"] == 512].copy()

    return df_best, t1_map, df_512, df_geral


# ==============================================================================
# CÁLCULOS MATEMÁTICOS: LEIS DE AMDAHL E GUSTAFSON
# ==============================================================================
def processar_escalabilidade_forte(df_best, p_max=4):
    """
    Aplica a Lei de Amdahl e métrica de Karp-Flatt:
      s = ((1 / S_p) - (1 / p)) / (1 - 1 / p)
      f = 1 - s
      S_max = 1 / s (quando s > 0)
    """
    df = df_best.copy()
    p = float(p_max)

    df["s_karp_flatt"] = (1.0 / df["Speedup"] - 1.0 / p) / (1.0 - 1.0 / p)
    # Trunca s para [0, 1] se estiver fora por desvios/desbalanceamentos patológicos
    df["s_karp_flatt_clamped"] = df["s_karp_flatt"].apply(lambda v: max(0.001, min(1.0, v)) if pd.notna(v) else np.nan)
    df["f_paralela"] = 1.0 - df["s_karp_flatt_clamped"]
    df["S_max_amdahl"] = 1.0 / df["s_karp_flatt_clamped"]

    return df


def processar_escalabilidade_fraca(df_best, t1_map):
    """
    Calcula as métricas de Escalabilidade Fraca (Gustafson-Barsis):
      Carga fixa por processador:
        Salto 1: Base 4K (p=1, 16.78M px) -> 8K (p=4, 67.11M px = 4 x 16.78M px)
        Salto 2: Base 8K (p=1, 67.11M px) -> 16K (p=4, 268.44M px = 4 x 67.11M px)
      Eficiência Fraca: E_weak = T1(W1) / Tp(4 * W1)
      Speedup Escalado Experimental: S_scaled = p * E_weak
      Speedup Teórico de Gustafson: S_Gustafson = p - s * (p - 1)
    """
    records = []
    p = 4.0

    cenarios = ["Visão Completa", "Zoom (Seahorse)"]
    estrategias = df_best["Estratégia"].unique()

    saltos = [
        {"nome": "4K (p=1) → 8K (p=4)", "base_res": 4096, "target_res": 8192, "px_proc": 16.78},
        {"nome": "8K (p=1) → 16K (p=4)", "base_res": 8192, "target_res": 16384, "px_proc": 67.11}
    ]

    for cenario in cenarios:
        for est in estrategias:
            sub = df_best[(df_best["Cenário"] == cenario) & (df_best["Estratégia"] == est)]

            for s in saltos:
                base_w = s["base_res"]
                target_w = s["target_res"]

                t1_base = t1_map.get((cenario, base_w))
                sub_target = sub[sub["Resolução"].str.contains(f"{target_w}")]

                if t1_base is not None and not sub_target.empty:
                    tp_target = float(sub_target["Tempo_s"].iloc[0])
                    sp_local = float(sub_target["Speedup"].iloc[0])

                    # Eficiência Fraca
                    e_weak = (t1_base / tp_target)
                    s_scaled = p * e_weak

                    # Fração serial no tempo paralelo medido (abordagem Gustafson-Barsis)
                    # S_gustafson = s + p*(1-s) = p - s*(p - 1) => s = (p - S_gustafson)/(p - 1)
                    s_gustafson = max(0.0, (p - s_scaled) / (p - 1.0)) if s_scaled <= p else 0.0
                    s_gustafson_teorico = p - s_gustafson * (p - 1.0)

                    records.append({
                        "Cenário": cenario,
                        "Estratégia": est,
                        "Salto_Carga": s["nome"],
                        "Base_W": base_w,
                        "Target_W": target_w,
                        "Pixels_Por_Proc_Mpx": s["px_proc"],
                        "Tempo_T1_Base_s": t1_base,
                        "Tempo_Tp_Alvo_s": tp_target,
                        "Eficiencia_Fraca_%": e_weak * 100.0,
                        "Speedup_Escalado": s_scaled,
                        "Fracao_Serial_Gustafson": s_gustafson,
                        "Speedup_Gustafson_Teorico": s_gustafson_teorico,
                        "Eficiencia_Fisica_Fraca_%": (e_weak * 2.0) * 100.0  # Base 2 núcleos físicos
                    })

    return pd.DataFrame(records)


# ==============================================================================
# PLOTAGEM DE GRÁFICOS (PNG 300 DPI + PDF VETORIAL)
# ==============================================================================

def annotar_pontos_sem_colisao(ax, pontos, x_is_category=False):
    """
    pontos: lista de tuplas (x_val, y_val, texto, cor)
    Agrupa por x_val, ordena por y_val e evita sobreposição de texto
    quando dois valores de y estão muito próximos.
    """
    from collections import defaultdict
    by_x = defaultdict(list)
    for x_val, y_val, texto, cor in pontos:
        by_x[x_val].append((y_val, texto, cor))

    for x_val, itens in by_x.items():
        itens_sorted = sorted(itens, key=lambda t: t[0])
        n = len(itens_sorted)
        offsets = [(0, 7) if x_is_category else (8, -3)] * n
        for i in range(1, n):
            if abs(itens_sorted[i][0] - itens_sorted[i-1][0]) < 0.12:
                if x_is_category:
                    offsets[i-1] = (-13, -11)
                    offsets[i] = (13, 8)
                else:
                    offsets[i-1] = (8, -11)
                    offsets[i] = (8, 6)

        for (y_val, texto, cor), (ox, oy) in zip(itens_sorted, offsets):
            ha_align = 'center' if (x_is_category and ox == 0) else ('left' if ox >= 0 else 'right')
            ax.annotate(texto, (x_val, y_val), textcoords="offset points",
                        xytext=(ox, oy), fontsize=8.5, fontweight='bold', color=cor,
                        ha=ha_align)


def plotar_escalabilidade_forte_amdahl(df_amdahl, dir_png, dir_pdf):
    """
    Gráfico 1: Escalabilidade Forte (Lei de Amdahl).
    Mostra o Speedup e a Eficiência em função do número de processadores p para
    tamanho fixo (16K - maior resolução estável), incluindo curvas teóricas de Amdahl.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]

    p_vals = np.linspace(1.0, 8.0, 100)
    estrategias_ordem = ['Estático Cíclico', 'Dinâmico Meia-Linha', 'Dinâmico Dedicado', 'Estático Bloco', 'Dinâmico Participativo']

    for cenario_nome, ax in cenarios:
        sub_c = df_amdahl[(df_amdahl["Cenário"] == cenario_nome) & (df_amdahl["Resolução"].str.contains("16K"))]

        # Linha Ideal Linear: Sp = p
        ax.plot(p_vals, p_vals, color=CORES_ESTRATEGIAS['Ideal Linear'], linestyle='--', linewidth=2.0,
                label='Speedup Ideal Linear ($S_p = p$)', zorder=2)

        # Curva de Amdahl de Referência para Fração Paralela de 90% e 95%
        s_ref_90 = 0.10
        s_ref_95 = 0.05
        ax.plot(p_vals, 1.0 / (s_ref_95 + (1.0 - s_ref_95) / p_vals), color='#9333ea', linestyle=':',
                linewidth=1.4, alpha=0.8, label=r'Amdahl Teórico ($f = 95\%$)')
        ax.plot(p_vals, 1.0 / (s_ref_90 + (1.0 - s_ref_90) / p_vals), color='#a855f7', linestyle='-.',
                linewidth=1.2, alpha=0.7, label=r'Amdahl Teórico ($f = 90\%$)')

        pontos_para_anotar = []

        # Plota cada estratégia MPI
        for est in estrategias_ordem:
            row = sub_c[sub_c["Estratégia"] == est]
            if row.empty:
                continue

            sp_4 = float(row["Speedup"].iloc[0])
            s_kf = float(row["s_karp_flatt_clamped"].iloc[0])
            s_max = float(row["S_max_amdahl"].iloc[0])
            cor = CORES_ESTRATEGIAS[est]
            mkr = MARCADORES[est]

            # Curva de Amdahl projetada pela fração serial real do algoritmo
            sp_amdahl_curve = 1.0 / (s_kf + (1.0 - s_kf) / p_vals)
            ax.plot(p_vals, sp_amdahl_curve, color=cor, linestyle='-', linewidth=1.8, alpha=0.6)

            # Pontos empíricos: p=1 (base serial) e p=4 (medido)
            ax.scatter([1.0], [1.0], color='#0f172a', s=70, marker='o', zorder=5)
            ax.scatter([4.0], [sp_4], color=cor, s=110, marker=mkr, edgecolor='black', linewidth=1.0, zorder=6,
                       label=rf'{est} ($S_4={sp_4:.2f}\times, s={s_kf*100:.1f}\%$)')

            pontos_para_anotar.append((4.0, sp_4, f"{sp_4:.2f}×", cor))

            # Linha horizontal sutil com o limite assintótico teórico S_max
            if s_max < 6.0 and s_max > 1.0:
                ax.axhline(s_max, color=cor, linestyle=':', alpha=0.35, linewidth=1.0)
                ax.text(7.9, s_max + 0.05, f"$S_{{max}}={s_max:.2f}$", color=cor, fontsize=7.5,
                        ha='right', va='bottom', style='italic')

        # Anotações sem colisão
        annotar_pontos_sem_colisao(ax, pontos_para_anotar, x_is_category=False)

        ax.set_title(f'Cenário: {cenario_nome} (Resolução Fixa: 16K)', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Unidades de Processamento ($p$)', fontsize=11, fontweight='bold')
        ax.set_xlim(0.8, 8.2)
        ax.set_ylim(0.0, 5.2)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
        ax.axvline(4.0, color='#64748b', linestyle=':', alpha=0.5, label='Capacidade Fís./HT ($p=4$)')

    axes[0].set_ylabel('Speedup de Amdahl ($S_p = T_1 / T_p$)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8.5)
    axes[0].legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8.5)

    plt.suptitle('Escalabilidade Forte (Lei de Amdahl): Speedup Real vs. Limite Assintótico ($W=16384^2$ Fixo)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    fig.savefig(os.path.join(dir_png, "mpi_escalabilidade_forte_amdahl.png"), dpi=300)
    fig.savefig(os.path.join(dir_pdf, "mpi_escalabilidade_forte_amdahl.pdf"))
    plt.close(fig)
    print(" Salvo: relatorios/png/mpi_escalabilidade_forte_amdahl.png")


def plotar_escalabilidade_forte_amdahl_resolucao(width: int, dir_png: str, dir_pdf: str):
    """
    Gráfico de Escalabilidade Forte Empírica por Resolução Fixa (4K, 8K ou 16K).
    Apresenta a curva empírica limpa com medições reais para p = 1, 2, 3 e 4 processadores,
    com escala ampliada, marcadores destacados e sem curvas teóricas de Amdahl.
    Utiliza posicionamento caso a caso refinado para garantir zero sobreposição entre linhas e rótulos.
    """
    tratados_path = resolve_path("data/dados_tratados_mpi.csv")
    if not os.path.exists(tratados_path):
        return

    df = pd.read_csv(tratados_path)

    def map_strat_name(code, sched):
        if code == 'ponto_a_ponto_v2': return 'Referência Serial (p=1)'
        if code == 'mpi_estatico' and sched == 'block': return 'Estático Bloco'
        if code == 'mpi_estatico' and sched == 'cyclic': return 'Estático Cíclico'
        if code == 'mpi_dinamico': return 'Dinâmico Dedicado'
        if code == 'mpi_dinamico_participativo': return 'Dinâmico Participativo'
        if code == 'mpi_dinamico_meia_linha': return 'Dinâmico Meia-Linha'
        return f'{code} ({sched})'

    df['Estratégia'] = df.apply(lambda r: map_strat_name(r['Code'], r['Schedule']), axis=1)

    sub = df[(df['WIDTH'] == width) & (df['Code'].str.startswith('mpi_'))].copy()
    if sub.empty:
        return

    idx = sub.groupby(['Scenario', 'Estratégia', 'Threads'])['Speedup_Tratado'].idxmax()
    best_res = sub.loc[idx]

    # Verifica se há dados além de p=4 para plotar a curva completa
    p_presentes = set(best_res['Threads'].unique())
    if len(p_presentes) < 2:
        return

    res_label = "4K" if width == 4096 else ("8K" if width == 8192 else ("16K" if width == 16384 else f"{width}"))

    # Mapeamento milimétrico caso a caso (Cenário, Estratégia, p) -> (dx, dy, ha, va)
    # Garante que cada caixa de texto fique em quadrantes livres sem interceptar nenhuma linha.
    OFFSETS_CASO_A_CASO = {
        # === CENÁRIO 1: VISÃO COMPLETA ===
        ('Visão Completa', 'Estático Bloco', 2): (-14, 12, 'right', 'bottom'),      # acima à esquerda da linha cinza
        ('Visão Completa', 'Estático Cíclico', 2): (14, -6, 'left', 'center'),       # no centro do V vazio entre reta azul subindo e cinza descendo
        ('Visão Completa', 'Dinâmico Meia-Linha', 2): (-14, 10, 'right', 'bottom'), # acima à esquerda de p=2 (acima do baseline 1.0)
        ('Visão Completa', 'Dinâmico Dedicado', 2): (-14, -10, 'right', 'top'),     # abaixo à esquerda de p=2
        ('Visão Completa', 'Dinâmico Participativo', 2): (0, -18, 'center', 'top'), # reto para baixo

        ('Visão Completa', 'Estático Cíclico', 3): (0, 14, 'center', 'bottom'),     # pico superior limpo
        ('Visão Completa', 'Dinâmico Dedicado', 3): (12, 10, 'left', 'bottom'),     # acima à direita
        ('Visão Completa', 'Dinâmico Meia-Linha', 3): (-14, 10, 'right', 'bottom'), # acima à esquerda
        ('Visão Completa', 'Estático Bloco', 3): (12, 10, 'left', 'bottom'),        # acima da linha horizontal cinza
        ('Visão Completa', 'Dinâmico Participativo', 3): (0, -18, 'center', 'top'), # reto para baixo

        ('Visão Completa', 'Estático Cíclico', 4): (12, 0, 'left', 'center'),
        ('Visão Completa', 'Dinâmico Meia-Linha', 4): (12, 10, 'left', 'bottom'),
        ('Visão Completa', 'Dinâmico Dedicado', 4): (12, -10, 'left', 'top'),
        ('Visão Completa', 'Estático Bloco', 4): (12, 0, 'left', 'center'),
        ('Visão Completa', 'Dinâmico Participativo', 4): (12, 0, 'left', 'center'),

        # === CENÁRIO 2: ZOOM (SEAHORSE) ===
        ('Zoom (Seahorse)', 'Estático Cíclico', 2): (14, -6, 'left', 'center'),       # no vão livre entre azul e cinza (abaixo da reta ideal)
        ('Zoom (Seahorse)', 'Estático Bloco', 2): (0, -18, 'center', 'top'),         # reto para baixo no vão vazio (0.94 até 1.63 é vazio!)
        ('Zoom (Seahorse)', 'Dinâmico Meia-Linha', 2): (-14, 10, 'right', 'bottom'), # acima à esquerda
        ('Zoom (Seahorse)', 'Dinâmico Dedicado', 2): (-14, -10, 'right', 'top'),     # abaixo à esquerda
        ('Zoom (Seahorse)', 'Dinâmico Participativo', 2): (0, -18, 'center', 'top'), # reto para baixo

        ('Zoom (Seahorse)', 'Estático Cíclico', 3): (0, 14, 'center', 'bottom'),     # topo absoluto
        ('Zoom (Seahorse)', 'Estático Bloco', 3): (0, -18, 'center', 'top'),         # reto para baixo dentro da tenda cinza, livre da reta azul descendo
        ('Zoom (Seahorse)', 'Dinâmico Dedicado', 3): (12, 10, 'left', 'bottom'),     # acima à direita
        ('Zoom (Seahorse)', 'Dinâmico Meia-Linha', 3): (-14, 10, 'right', 'bottom'), # acima à esquerda
        ('Zoom (Seahorse)', 'Dinâmico Participativo', 3): (0, -18, 'center', 'top'), # reto para baixo

        ('Zoom (Seahorse)', 'Estático Cíclico', 4): (12, 8, 'left', 'bottom'),
        ('Zoom (Seahorse)', 'Estático Bloco', 4): (12, -8, 'left', 'top'),
        ('Zoom (Seahorse)', 'Dinâmico Dedicado', 4): (12, 8, 'left', 'bottom'),
        ('Zoom (Seahorse)', 'Dinâmico Meia-Linha', 4): (12, -8, 'left', 'top'),
        ('Zoom (Seahorse)', 'Dinâmico Participativo', 4): (12, 0, 'left', 'center'),

        # === AJUSTES ESPECÍFICOS PARA 16K ===
        ('16K', 'Visão Completa', 'Dinâmico Dedicado', 4): (12, 6, 'left', 'bottom'),     # 1.45x fica acima
        ('16K', 'Visão Completa', 'Dinâmico Meia-Linha', 4): (12, -8, 'left', 'top'),     # 1.43x fica abaixo
        ('16K', 'Zoom (Seahorse)', 'Dinâmico Meia-Linha', 4): (12, 8, 'left', 'bottom'),  # 1.72x fica acima
        ('16K', 'Zoom (Seahorse)', 'Estático Bloco', 4): (12, -8, 'left', 'top'),         # 1.68x fica abaixo
        ('16K', 'Zoom (Seahorse)', 'Dinâmico Dedicado', 4): (12, 0, 'left', 'center'),    # 1.37x centrado
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), sharey=True)
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]

    estrategias_ordem = ['Estático Cíclico', 'Dinâmico Meia-Linha', 'Dinâmico Dedicado', 'Estático Bloco', 'Dinâmico Participativo']

    for cenario_nome, ax in cenarios:
        # Linha Ideal Linear (referência pura)
        p_ideal = np.linspace(1.0, 3.0, 100)
        ax.plot(p_ideal, p_ideal, color='#ef4444', linestyle='--', linewidth=1.8, alpha=0.65,
                label='Speedup Linear Ideal (Sp = p)', zorder=2)

        # Marcador baseline serial
        ax.scatter([1.0], [1.0], color='#0f172a', s=130, marker='o', zorder=5)

        for est in estrategias_ordem:
            sub_est = best_res[(best_res['Scenario'] == cenario_nome) & (best_res['Estratégia'] == est)].sort_values('Threads')
            if sub_est.empty:
                continue

            p_medidos = [1] + sub_est['Threads'].tolist()
            sp_medidos = [1.0] + sub_est['Speedup_Tratado'].tolist()

            cor = CORES_ESTRATEGIAS[est]
            mkr = MARCADORES[est]
            sp4 = sp_medidos[-1]

            # Curva empírica conectando medições reais
            ax.plot(p_medidos, sp_medidos, color=cor, linestyle='-', linewidth=2.8, alpha=0.9, zorder=4,
                    label=f'{est} (S4 = {sp4:.2f}x)')

            # Marcadores ampliados destacados
            ax.scatter(p_medidos[1:], sp_medidos[1:], color=cor, s=160, marker=mkr, edgecolor='#0f172a', linewidth=1.2, zorder=6)

            for p_pt, sp_pt in zip(p_medidos[1:], sp_medidos[1:]):
                key_res = (res_label, cenario_nome, est, p_pt)
                key_generic = (cenario_nome, est, p_pt)
                dx, dy, ha_v, va_v = OFFSETS_CASO_A_CASO.get(key_res, OFFSETS_CASO_A_CASO.get(key_generic, (10, 0, 'left', 'center')))
                ax.annotate(
                    f"{sp_pt:.2f}x",
                    (p_pt, sp_pt),
                    textcoords='offset points',
                    xytext=(dx, dy),
                    fontsize=9.2,
                    fontweight='bold',
                    color=cor,
                    va=va_v,
                    ha=ha_v,
                    zorder=10,
                    bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=cor, alpha=1.0, linewidth=0.9)
                )

        ax.set_title(f'Cenário: {cenario_nome} (Resolução Fixa: {res_label})', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Unidades de Processamento (p)', fontsize=12, fontweight='bold')
        ax.set_xlim(0.85, 4.35)
        ax.set_ylim(0.0, 3.05)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.axvline(2.0, color='#0284c7', linestyle='--', alpha=0.6, linewidth=1.2, label='Núcleos Físicos (p = 2)')
        ax.axvline(4.0, color='#64748b', linestyle=':', alpha=0.6, linewidth=1.2, label='Capacidade Fís./HT (p = 4)')

    axes[0].set_ylabel('Speedup Empírico Real (Sp = T1 / Tp)', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=9.2)
    axes[1].legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=9.2)

    plt.suptitle(f'Escalabilidade Forte Empírica (Resolução {res_label} Fixa): Speedup Real por Processos (p = 1, 2, 3, 4)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_name = f"mpi_escalabilidade_forte_amdahl_{res_label.lower()}"
    fig.savefig(os.path.join(dir_png, f"{out_name}.png"), dpi=300)
    fig.savefig(os.path.join(dir_pdf, f"{out_name}.pdf"))
    plt.close(fig)
    print(f" Salvo: relatorios/png/{out_name}.png")


def plotar_escalabilidade_forte_amdahl_4k(dir_png: str, dir_pdf: str):
    """Gera o gráfico de escalabilidade forte para 4K."""
    plotar_escalabilidade_forte_amdahl_resolucao(4096, dir_png, dir_pdf)


def plotar_escalabilidade_forte_amdahl_8k(dir_png: str, dir_pdf: str):
    """Gera o gráfico de escalabilidade forte para 8K."""
    plotar_escalabilidade_forte_amdahl_resolucao(8192, dir_png, dir_pdf)


def plotar_escalabilidade_forte_amdahl_16k(dir_png: str, dir_pdf: str):
    """Gera o gráfico de escalabilidade forte para 16K."""
    plotar_escalabilidade_forte_amdahl_resolucao(16384, dir_png, dir_pdf)


def plotar_gustafson_tempo_salto(salto_id: int, dir_png: str, dir_pdf: str):
    """
    Forma 1 (Lei de Gustafson): Gráfico de Linhas de TEMPO DE EXECUÇÃO por NÚMERO DE PROCESSADORES (p = 1, 2, 3, 4).
    Mapeia a expansão do problema proporcional ao número de nós:
      - Salto 1 (salto_id=1): Base 4K (p=1, 16.78M px) -> 8K (p=2, 3, 4)
      - Salto 2 (salto_id=2): Base 8K (p=1, 67.11M px) -> 16K (p=2, 3, 4)
    Apresenta a linha de referência ideal constante de Gustafson (T = T1_base), destacando
    os desvios causados por sobrecarga de comunicação e desbalanceamento de carga.
    """
    tratados_path = resolve_path("data/dados_tratados_mpi.csv")
    if not os.path.exists(tratados_path):
        return

    df = pd.read_csv(tratados_path)

    def map_strat_name(code, sched):
        if code == 'ponto_a_ponto_v2': return 'Referência Serial (p=1)'
        if code == 'mpi_estatico' and sched == 'block': return 'Estático Bloco'
        if code == 'mpi_estatico' and sched == 'cyclic': return 'Estático Cíclico'
        if code == 'mpi_dinamico': return 'Dinâmico Dedicado'
        if code == 'mpi_dinamico_participativo': return 'Dinâmico Participativo'
        if code == 'mpi_dinamico_meia_linha': return 'Dinâmico Meia-Linha'
        return f'{code} ({sched})'

    df['Estratégia'] = df.apply(lambda r: map_strat_name(r['Code'], r['Schedule']), axis=1)

    if salto_id == 1:
        base_w, target_w = 4096, 8192
        salto_nome = "Salto 1: 4K (p=1) → 8K (p=2, 3, 4)"
        px_proc = "16.8M px/processo"
        fname_base = "mpi_gustafson_tempo_salto1_4k_para_8k"
    else:
        base_w, target_w = 8192, 16384
        salto_nome = "Salto 2: 8K (p=1) → 16K (p=2, 3, 4)"
        px_proc = "67.1M px/processo"
        fname_base = "mpi_gustafson_tempo_salto2_8k_para_16k"

    OFFSETS_TEMPO = {
        # SALTO 1 - VISÃO COMPLETA
        (1, "Visão Completa", "Estático Cíclico", 2): (12, -6, "left", "center"),
        (1, "Visão Completa", "Estático Bloco", 2): (-14, 8, "right", "bottom"),
        (1, "Visão Completa", "Dinâmico Meia-Linha", 2): (12, 10, "left", "bottom"),
        (1, "Visão Completa", "Dinâmico Dedicado", 2): (-14, 0, "right", "center"),
        (1, "Visão Completa", "Dinâmico Participativo", 2): (12, -10, "left", "top"),

        (1, "Visão Completa", "Estático Cíclico", 3): (0, -18, "center", "top"),
        (1, "Visão Completa", "Dinâmico Meia-Linha", 3): (-14, 8, "right", "bottom"),
        (1, "Visão Completa", "Dinâmico Dedicado", 3): (12, 8, "left", "bottom"),
        (1, "Visão Completa", "Estático Bloco", 3): (16, 0, "left", "center"),
        (1, "Visão Completa", "Dinâmico Participativo", 3): (12, 0, "left", "center"),

        (1, "Visão Completa", "Estático Cíclico", 4): (12, 0, "left", "center"),
        (1, "Visão Completa", "Dinâmico Dedicado", 4): (14, 8, "left", "bottom"),
        (1, "Visão Completa", "Dinâmico Meia-Linha", 4): (14, -8, "left", "top"),
        (1, "Visão Completa", "Estático Bloco", 4): (12, 0, "left", "center"),
        (1, "Visão Completa", "Dinâmico Participativo", 4): (12, 0, "left", "center"),

        # SALTO 1 - ZOOM SEAHORSE
        (1, "Zoom (Seahorse)", "Estático Cíclico", 2): (-14, -8, "right", "top"),
        (1, "Zoom (Seahorse)", "Estático Bloco", 2): (12, 8, "left", "bottom"),
        (1, "Zoom (Seahorse)", "Dinâmico Participativo", 2): (12, -10, "left", "top"),
        (1, "Zoom (Seahorse)", "Dinâmico Meia-Linha", 2): (-14, 0, "right", "center"),
        (1, "Zoom (Seahorse)", "Dinâmico Dedicado", 2): (12, 10, "left", "bottom"),

        (1, "Zoom (Seahorse)", "Estático Cíclico", 3): (0, -18, "center", "top"),
        (1, "Zoom (Seahorse)", "Estático Bloco", 3): (16, 0, "left", "center"),
        (1, "Zoom (Seahorse)", "Dinâmico Meia-Linha", 3): (-14, 8, "right", "bottom"),
        (1, "Zoom (Seahorse)", "Dinâmico Dedicado", 3): (12, 8, "left", "bottom"),
        (1, "Zoom (Seahorse)", "Dinâmico Participativo", 3): (12, 0, "left", "center"),

        (1, "Zoom (Seahorse)", "Estático Cíclico", 4): (12, -6, "left", "top"),
        (1, "Zoom (Seahorse)", "Estático Bloco", 4): (14, -8, "left", "top"),
        (1, "Zoom (Seahorse)", "Dinâmico Dedicado", 4): (14, 10, "left", "bottom"),
        (1, "Zoom (Seahorse)", "Dinâmico Meia-Linha", 4): (-14, 0, "right", "center"),
        (1, "Zoom (Seahorse)", "Dinâmico Participativo", 4): (12, 0, "left", "center"),

        # SALTO 2 - VISÃO COMPLETA
        (2, "Visão Completa", "Estático Bloco", 2): (-14, 8, "right", "bottom"),
        (2, "Visão Completa", "Estático Cíclico", 2): (12, -6, "left", "center"),
        (2, "Visão Completa", "Dinâmico Meia-Linha", 2): (12, 10, "left", "bottom"),
        (2, "Visão Completa", "Dinâmico Dedicado", 2): (-14, 0, "right", "center"),
        (2, "Visão Completa", "Dinâmico Participativo", 2): (12, -10, "left", "top"),

        (2, "Visão Completa", "Estático Cíclico", 3): (0, -18, "center", "top"),
        (2, "Visão Completa", "Dinâmico Meia-Linha", 3): (-14, 8, "right", "bottom"),
        (2, "Visão Completa", "Dinâmico Dedicado", 3): (12, 8, "left", "bottom"),
        (2, "Visão Completa", "Estático Bloco", 3): (16, 0, "left", "center"),
        (2, "Visão Completa", "Dinâmico Participativo", 3): (12, 0, "left", "center"),

        (2, "Visão Completa", "Estático Cíclico", 4): (12, 0, "left", "center"),
        (2, "Visão Completa", "Dinâmico Dedicado", 4): (12, 8, "left", "bottom"),
        (2, "Visão Completa", "Dinâmico Meia-Linha", 4): (12, -8, "left", "top"),
        (2, "Visão Completa", "Estático Bloco", 4): (12, 0, "left", "center"),
        (2, "Visão Completa", "Dinâmico Participativo", 4): (12, 0, "left", "center"),

        # SALTO 2 - ZOOM SEAHORSE
        (2, "Zoom (Seahorse)", "Estático Cíclico", 2): (-14, -8, "right", "top"),
        (2, "Zoom (Seahorse)", "Estático Bloco", 2): (12, 8, "left", "bottom"),
        (2, "Zoom (Seahorse)", "Dinâmico Participativo", 2): (12, -10, "left", "top"),
        (2, "Zoom (Seahorse)", "Dinâmico Meia-Linha", 2): (-14, 0, "right", "center"),
        (2, "Zoom (Seahorse)", "Dinâmico Dedicado", 2): (12, 10, "left", "bottom"),

        (2, "Zoom (Seahorse)", "Estático Cíclico", 3): (0, -18, "center", "top"),
        (2, "Zoom (Seahorse)", "Estático Bloco", 3): (-14, -8, "right", "top"),
        (2, "Zoom (Seahorse)", "Dinâmico Dedicado", 3): (12, 8, "left", "bottom"),
        (2, "Zoom (Seahorse)", "Dinâmico Meia-Linha", 3): (-14, 8, "right", "bottom"),
        (2, "Zoom (Seahorse)", "Dinâmico Participativo", 3): (12, 0, "left", "center"),

        (2, "Zoom (Seahorse)", "Estático Cíclico", 4): (14, -8, "left", "top"),
        (2, "Zoom (Seahorse)", "Dinâmico Meia-Linha", 4): (14, 8, "left", "bottom"),
        (2, "Zoom (Seahorse)", "Estático Bloco", 4): (-14, 8, "right", "bottom"),
        (2, "Zoom (Seahorse)", "Dinâmico Dedicado", 4): (12, 0, "left", "center"),
        (2, "Zoom (Seahorse)", "Dinâmico Participativo", 4): (12, 0, "left", "center"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]
    estrategias_ordem = ['Estático Cíclico', 'Dinâmico Meia-Linha', 'Dinâmico Dedicado', 'Estático Bloco', 'Dinâmico Participativo']

    for cenario_nome, ax in cenarios:
        # T1 da resolução base
        t1_row = df[(df['WIDTH'] == base_w) & (df['Code'] == 'ponto_a_ponto_v2') & (df['Scenario'] == cenario_nome)]
        if t1_row.empty:
            continue
        t1_base = float(t1_row['Tempo_Tratado'].iloc[0])

        # Linha horizontal de Tempo Ideal de Gustafson (T = T1_base constante)
        ax.axhline(t1_base, color='#ef4444', linestyle='--', linewidth=1.8, alpha=0.85,
                   label=f'Tempo Constante Ideal Gustafson (T = {t1_base:.2f}s)', zorder=2)

        # Marcador baseline serial
        ax.scatter([1.0], [t1_base], color='#0f172a', s=140, marker='o', zorder=5)
        ax.annotate(f"T1 Base\n{t1_base:.2f}s", (1.0, t1_base),
                    xytext=(-12, 10), textcoords='offset points',
                    fontsize=9.2, fontweight='bold', color='#0f172a',
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='#0f172a', linewidth=0.9, alpha=1.0, zorder=10))

        sub_target = df[(df['WIDTH'] == target_w) & (df['Scenario'] == cenario_nome) & (df['Code'].str.startswith('mpi_'))]
        max_y_local = t1_base

        for est in estrategias_ordem:
            sub_est = sub_target[sub_target['Estratégia'] == est]
            if sub_est.empty:
                continue

            idx_best = sub_est.groupby('Threads')['Tempo_Tratado'].idxmin()
            best_points = sub_est.loc[idx_best].sort_values('Threads')

            p_list = [1] + best_points['Threads'].tolist()
            t_list = [t1_base] + best_points['Tempo_Tratado'].tolist()
            max_y_local = max(max_y_local, max(t_list))

            cor = CORES_ESTRATEGIAS[est]
            mkr = MARCADORES[est]
            t4_val = t_list[-1]
            e_weak_4 = (t1_base / t4_val) * 100.0

            # Curva de tempo
            ax.plot(p_list, t_list, color=cor, linestyle='-', linewidth=2.8, alpha=0.9, zorder=4,
                    label=f'{est} (T4={t4_val:.1f}s, E={e_weak_4:.1f}%)')
            ax.scatter(p_list[1:], t_list[1:], color=cor, s=160, marker=mkr, edgecolor='#0f172a', linewidth=1.2, zorder=6)

            for p_pt, t_pt in zip(p_list[1:], t_list[1:]):
                key = (salto_id, cenario_nome, est, p_pt)
                dx, dy, ha_v, va_v = OFFSETS_TEMPO.get(key, (10, 0, 'left', 'center'))

                if p_pt == 4:
                    txt = f"{t_pt:.2f}s\n(E={e_weak_4:.1f}%)"
                else:
                    txt = f"{t_pt:.2f}s"

                ax.annotate(
                    txt,
                    (p_pt, t_pt),
                    textcoords='offset points',
                    xytext=(dx, dy),
                    fontsize=8.8,
                    fontweight='bold',
                    color=cor,
                    va=va_v,
                    ha=ha_v,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=cor, linewidth=0.9, alpha=1.0, zorder=10)
                )

        # Linhas de referência de hardware
        ax.axvline(2.0, color='#0284c7', linestyle='--', linewidth=1.2, alpha=0.6, label='Núcleos Físicos (p = 2)', zorder=2)
        ax.axvline(4.0, color='#64748b', linestyle=':', linewidth=1.2, alpha=0.6, label='Capacidade Fís./HT (p = 4)', zorder=2)

        ax.set_title(f'Cenário: {cenario_nome} [{salto_nome}]', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Unidades de Processamento (p)', fontsize=11, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['1\n(Serial Base)', '2\n(Núcleos Fís.)', '3\n(SMT/HT)', '4\n(Capacidade Máx.)'], fontsize=9.5)
        ax.set_xlim(0.65, 4.75)
        ax.set_ylim(0, max_y_local * 1.15)
        ax.grid(True, linestyle='--', alpha=0.7)

    axes[0].set_ylabel('Tempo de Execução Real (segundos)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=8.2)
    axes[0].legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=8.2)

    plt.suptitle(f'Escalabilidade Fraca (Lei de Gustafson): Tempo de Execução vs Processadores\n[{salto_nome} - {px_proc}]',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    fig.savefig(os.path.join(dir_png, f"{fname_base}.png"), dpi=300)
    fig.savefig(os.path.join(dir_pdf, f"{fname_base}.pdf"))
    plt.close(fig)
    print(f" Salvo: relatorios/png/{fname_base}.png")


def plotar_gustafson_tempo_salto1(dir_png: str, dir_pdf: str):
    """Gera o gráfico de tempo de Gustafson para o Salto 1 (4K -> 8K)."""
    plotar_gustafson_tempo_salto(1, dir_png, dir_pdf)


def plotar_gustafson_tempo_salto2(dir_png: str, dir_pdf: str):
    """Gera o gráfico de tempo de Gustafson para o Salto 2 (8K -> 16K)."""
    plotar_gustafson_tempo_salto(2, dir_png, dir_pdf)


def plotar_escalabilidade_fraca_gustafson(df_gustafson, dir_png, dir_pdf):
    """
    Forma 2 (Lei de Gustafson): Gráfico de Linhas de EFICIÊNCIA FRACA (%) entre os Saltos.
    Substitui o antigo gráfico de barras por uma representação de linhas limpa e contínua.
    Mostra a Eficiência Fraca (E_weak = T1 / Tp %) e o Speedup Escalado nos dois saltos
    de carga proporcional (4K -> 8K e 8K -> 16K).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]

    estrategias_ordem = ['Estático Cíclico', 'Dinâmico Meia-Linha', 'Dinâmico Dedicado', 'Estático Bloco', 'Dinâmico Participativo']
    saltos = ["4K (p=1) → 8K (p=4)", "8K (p=1) → 16K (p=4)"]
    x_indices = [0.0, 1.0]

    # Offsets refinados caso a caso garantindo espaçamento vertical perfeito na escala ampliada
    OFFSETS_EF = {
        # VISÃO COMPLETA
        ("Visão Completa", "Estático Cíclico", 0): (-14, 6, "right", "bottom"),
        ("Visão Completa", "Estático Cíclico", 1): (14, 0, "left", "center"),
        ("Visão Completa", "Dinâmico Meia-Linha", 0): (-14, 6, "right", "bottom"),
        ("Visão Completa", "Dinâmico Meia-Linha", 1): (14, -8, "left", "top"),
        ("Visão Completa", "Dinâmico Dedicado", 0): (-14, -8, "right", "top"),
        ("Visão Completa", "Dinâmico Dedicado", 1): (14, 8, "left", "bottom"),
        ("Visão Completa", "Estático Bloco", 0): (-14, 0, "right", "center"),
        ("Visão Completa", "Estático Bloco", 1): (14, 0, "left", "center"),
        ("Visão Completa", "Dinâmico Participativo", 0): (-14, 0, "right", "center"),
        ("Visão Completa", "Dinâmico Participativo", 1): (14, 0, "left", "center"),

        # ZOOM SEAHORSE
        ("Zoom (Seahorse)", "Estático Cíclico", 0): (-14, 8, "right", "bottom"),
        ("Zoom (Seahorse)", "Estático Cíclico", 1): (14, 10, "left", "bottom"),
        ("Zoom (Seahorse)", "Estático Bloco", 0): (-14, 8, "right", "bottom"),
        ("Zoom (Seahorse)", "Estático Bloco", 1): (14, -8, "left", "top"),
        ("Zoom (Seahorse)", "Dinâmico Meia-Linha", 0): (14, -14, "left", "top"),
        ("Zoom (Seahorse)", "Dinâmico Meia-Linha", 1): (14, 8, "left", "bottom"),
        ("Zoom (Seahorse)", "Dinâmico Dedicado", 0): (-14, -8, "right", "top"),
        ("Zoom (Seahorse)", "Dinâmico Dedicado", 1): (14, -8, "left", "top"),
        ("Zoom (Seahorse)", "Dinâmico Participativo", 0): (-14, 0, "right", "center"),
        ("Zoom (Seahorse)", "Dinâmico Participativo", 1): (14, 0, "left", "center"),
    }

    for cenario_nome, ax in cenarios:
        sub_c = df_gustafson[df_gustafson["Cenário"] == cenario_nome]

        # Mantém apenas o patamar físico de 2 núcleos (50%), sem a linha ideal de 100% para ampliar a escala
        ax.axhline(50.0, color='#0284c7', linestyle=':', linewidth=1.5,
                   label='Patamar 50% (S4 = 2.00x - Limite 2 Núcleos Físicos)')

        for est in estrategias_ordem:
            sub_est = sub_c[sub_c["Estratégia"] == est]
            ef_vals = []
            sp_vals = []
            for s in saltos:
                val = sub_est[sub_est["Salto_Carga"] == s]
                ef_vals.append(val["Eficiencia_Fraca_%"].iloc[0] if not val.empty else 0.0)
                sp_vals.append(val["Speedup_Escalado"].iloc[0] if not val.empty else 0.0)

            cor = CORES_ESTRATEGIAS[est]
            mkr = MARCADORES[est]

            # Traça a linha contínua entre os saltos
            ax.plot(x_indices, ef_vals, color=cor, linestyle='-', linewidth=2.8, alpha=0.9, zorder=4,
                    label=f'{est} (S2={sp_vals[1]:.2f}x, E2={ef_vals[1]:.1f}%)')
            ax.scatter(x_indices, ef_vals, color=cor, s=160, marker=mkr, edgecolor='#0f172a', linewidth=1.2, zorder=6)

            for idx_pt, (x_pos, ef_val, sp_val) in enumerate(zip(x_indices, ef_vals, sp_vals)):
                key = (cenario_nome, est, idx_pt)
                dx, dy, ha_v, va_v = OFFSETS_EF.get(key, (-12 if idx_pt == 0 else 12, 0, 'right' if idx_pt == 0 else 'left', 'center'))

                ax.annotate(
                    f"{ef_val:.1f}%\n(Sp={sp_val:.2f}x)",
                    (x_pos, ef_val),
                    textcoords='offset points',
                    xytext=(dx, dy),
                    fontsize=9.0,
                    fontweight='bold',
                    color=cor,
                    va=va_v,
                    ha=ha_v,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=cor, linewidth=0.9, alpha=1.0, zorder=10)
                )

        ax.set_title(f'Cenário: {cenario_nome}', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Expansão de Carga Proporcional (W ∝ p)', fontsize=11, fontweight='bold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels([
            'Salto 1: 4K (p=1) → 8K (p=4)\n[16.8M px/processo]',
            'Salto 2: 8K (p=1) → 16K (p=4)\n[67.1M px/processo]'
        ], fontsize=9.5)
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(0, 75)
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=8.4)

    axes[0].set_ylabel('Eficiência de Escalabilidade Fraca (E_weak = T1 / Tp %)', fontsize=12, fontweight='bold')

    plt.suptitle('Escalabilidade Fraca (Lei de Gustafson): Evolução da Eficiência Fraca sob Carga Constante\n[Comparativo entre Saltos em Gráfico de Linhas - Escala Ampliada]',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    fig.savefig(os.path.join(dir_png, "mpi_escalabilidade_fraca_gustafson.png"), dpi=300)
    fig.savefig(os.path.join(dir_pdf, "mpi_escalabilidade_fraca_gustafson.pdf"))
    fig.savefig(os.path.join(dir_png, "mpi_gustafson_eficiencia_fraca_linhas.png"), dpi=300)
    fig.savefig(os.path.join(dir_pdf, "mpi_gustafson_eficiencia_fraca_linhas.pdf"))
    plt.close(fig)
    print(" Salvo: relatorios/png/mpi_escalabilidade_fraca_gustafson.png")
    print(" Salvo: relatorios/png/mpi_gustafson_eficiencia_fraca_linhas.png")


def plotar_gustafson_speedup_escalado_linhas(df_gustafson, dir_png, dir_pdf):
    """
    Gráfico de Linhas de SPEEDUP ESCALADO DE GUSTAFSON (S_scaled = p * E_weak).
    Apresenta o Speedup Escalado nos dois saltos de carga proporcional,
    comparando diretamente com o Speedup Ideal de Gustafson (S = 4.00x)
    e com a barreira física de 2 núcleos (S = 2.00x).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]

    estrategias_ordem = ['Estático Cíclico', 'Dinâmico Meia-Linha', 'Dinâmico Dedicado', 'Estático Bloco', 'Dinâmico Participativo']
    saltos = ["4K (p=1) → 8K (p=4)", "8K (p=1) → 16K (p=4)"]
    x_indices = [0.0, 1.0]

    OFFSETS_SP = {
        ("Visão Completa", "Estático Cíclico", 0): (-14, 8, "right", "bottom"),
        ("Visão Completa", "Estático Cíclico", 1): (14, 8, "left", "bottom"),
        ("Visão Completa", "Dinâmico Meia-Linha", 0): (-14, 6, "right", "bottom"),
        ("Visão Completa", "Dinâmico Meia-Linha", 1): (14, -6, "left", "top"),
        ("Visão Completa", "Dinâmico Dedicado", 0): (-14, -8, "right", "top"),
        ("Visão Completa", "Dinâmico Dedicado", 1): (14, 6, "left", "bottom"),
        ("Visão Completa", "Estático Bloco", 0): (-14, 0, "right", "center"),
        ("Visão Completa", "Estático Bloco", 1): (14, 0, "left", "center"),
        ("Visão Completa", "Dinâmico Participativo", 0): (-14, 0, "right", "center"),
        ("Visão Completa", "Dinâmico Participativo", 1): (14, 0, "left", "center"),

        ("Zoom (Seahorse)", "Estático Cíclico", 0): (-14, 8, "right", "bottom"),
        ("Zoom (Seahorse)", "Estático Cíclico", 1): (14, 12, "left", "bottom"),
        ("Zoom (Seahorse)", "Estático Bloco", 0): (-14, 0, "right", "center"),
        ("Zoom (Seahorse)", "Estático Bloco", 1): (14, -10, "left", "top"),
        ("Zoom (Seahorse)", "Dinâmico Dedicado", 0): (-14, 8, "right", "bottom"),
        ("Zoom (Seahorse)", "Dinâmico Dedicado", 1): (14, 0, "left", "center"),
        ("Zoom (Seahorse)", "Dinâmico Meia-Linha", 0): (-14, -8, "right", "top"),
        ("Zoom (Seahorse)", "Dinâmico Meia-Linha", 1): (14, 4, "left", "bottom"),
        ("Zoom (Seahorse)", "Dinâmico Participativo", 0): (-14, 0, "right", "center"),
        ("Zoom (Seahorse)", "Dinâmico Participativo", 1): (14, 0, "left", "center"),
    }

    for cenario_nome, ax in cenarios:
        sub_c = df_gustafson[df_gustafson["Cenário"] == cenario_nome]

        ax.axhline(4.0, color='#ef4444', linestyle='--', linewidth=1.8, alpha=0.85,
                   label='Speedup Ideal Gustafson (S = 4.00x - 100%)')
        ax.axhline(2.0, color='#0284c7', linestyle=':', linewidth=1.5,
                   label='Patamar 2 Núcleos Físicos (S = 2.00x - 50%)')

        for est in estrategias_ordem:
            sub_est = sub_c[sub_c["Estratégia"] == est]
            sp_vals = []
            ef_vals = []
            for s in saltos:
                val = sub_est[sub_est["Salto_Carga"] == s]
                sp_vals.append(val["Speedup_Escalado"].iloc[0] if not val.empty else 0.0)
                ef_vals.append(val["Eficiencia_Fraca_%"].iloc[0] if not val.empty else 0.0)

            cor = CORES_ESTRATEGIAS[est]
            mkr = MARCADORES[est]

            ax.plot(x_indices, sp_vals, color=cor, linestyle='-', linewidth=2.8, alpha=0.9, zorder=4,
                    label=f'{est} (S2={sp_vals[1]:.2f}x)')
            ax.scatter(x_indices, sp_vals, color=cor, s=160, marker=mkr, edgecolor='#0f172a', linewidth=1.2, zorder=6)

            for idx_pt, (x_pos, sp_val, ef_val) in enumerate(zip(x_indices, sp_vals, ef_vals)):
                key = (cenario_nome, est, idx_pt)
                dx, dy, ha_v, va_v = OFFSETS_SP.get(key, (-12 if idx_pt == 0 else 12, 0, 'right' if idx_pt == 0 else 'left', 'center'))

                ax.annotate(
                    f"{sp_val:.2f}x\n({ef_val:.1f}%)",
                    (x_pos, sp_val),
                    textcoords='offset points',
                    xytext=(dx, dy),
                    fontsize=8.8,
                    fontweight='bold',
                    color=cor,
                    va=va_v,
                    ha=ha_v,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=cor, linewidth=0.9, alpha=1.0, zorder=10)
                )

        ax.set_title(f'Cenário: {cenario_nome}', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Expansão de Carga Proporcional (W ∝ p)', fontsize=11, fontweight='bold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels([
            'Salto 1: 4K (p=1) → 8K (p=4)\n[16.8M px/processo]',
            'Salto 2: 8K (p=1) → 16K (p=4)\n[67.1M px/processo]'
        ], fontsize=9.5)
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(0, 4.75)
        ax.grid(True, linestyle='--', alpha=0.7)

    axes[0].set_ylabel('Speedup Escalado de Gustafson (S_scaled = p * E_weak)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=8.2)
    axes[0].legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=8.2)

    plt.suptitle('Escalabilidade Fraca (Lei de Gustafson): Speedup Escalado Real por Expansão de Carga',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    fig.savefig(os.path.join(dir_png, "mpi_gustafson_speedup_escalado_linhas.png"), dpi=300)
    fig.savefig(os.path.join(dir_pdf, "mpi_gustafson_speedup_escalado_linhas.pdf"))
    plt.close(fig)
    print(" Salvo: relatorios/png/mpi_gustafson_speedup_escalado_linhas.png")


def plotar_gustafson_tempo_consolidado(dir_png: str, dir_pdf: str):
    """
    Gera um painel comparativo 2x2 consolidando o tempo por processador para os dois saltos e dois cenários.
    """
    tratados_path = resolve_path("data/dados_tratados_mpi.csv")
    if not os.path.exists(tratados_path):
        return

    df = pd.read_csv(tratados_path)

    def map_strat_name(code, sched):
        if code == 'ponto_a_ponto_v2': return 'Referência Serial (p=1)'
        if code == 'mpi_estatico' and sched == 'block': return 'Estático Bloco'
        if code == 'mpi_estatico' and sched == 'cyclic': return 'Estático Cíclico'
        if code == 'mpi_dinamico': return 'Dinâmico Dedicado'
        if code == 'mpi_dinamico_participativo': return 'Dinâmico Participativo'
        if code == 'mpi_dinamico_meia_linha': return 'Dinâmico Meia-Linha'
        return f'{code} ({sched})'

    df['Estratégia'] = df.apply(lambda r: map_strat_name(r['Code'], r['Schedule']), axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    saltos_info = [
        (1, 4096, 8192, "Salto 1: 4K (p=1) → 8K (p=2, 3, 4)", "16.8M px/proc", axes[0]),
        (2, 8192, 16384, "Salto 2: 8K (p=1) → 16K (p=2, 3, 4)", "67.1M px/proc", axes[1]),
    ]
    estrategias_ordem = ['Estático Cíclico', 'Dinâmico Meia-Linha', 'Dinâmico Dedicado', 'Estático Bloco', 'Dinâmico Participativo']

    for salto_id, base_w, target_w, salto_nome, px_proc, row_axes in saltos_info:
        for cenario_nome, ax in [("Visão Completa", row_axes[0]), ("Zoom (Seahorse)", row_axes[1])]:
            t1_row = df[(df['WIDTH'] == base_w) & (df['Code'] == 'ponto_a_ponto_v2') & (df['Scenario'] == cenario_nome)]
            if t1_row.empty: continue
            t1_base = float(t1_row['Tempo_Tratado'].iloc[0])

            ax.axhline(t1_base, color='#ef4444', linestyle='--', linewidth=1.6, alpha=0.85,
                       label=f'Tempo Ideal Gustafson (T = {t1_base:.1f}s)', zorder=2)
            ax.scatter([1.0], [t1_base], color='#0f172a', s=120, marker='o', zorder=5)

            sub_target = df[(df['WIDTH'] == target_w) & (df['Scenario'] == cenario_nome) & (df['Code'].str.startswith('mpi_'))]
            max_y_local = t1_base

            for est in estrategias_ordem:
                sub_est = sub_target[sub_target['Estratégia'] == est]
                if sub_est.empty: continue
                idx_best = sub_est.groupby('Threads')['Tempo_Tratado'].idxmin()
                best_pts = sub_est.loc[idx_best].sort_values('Threads')

                p_list = [1] + best_pts['Threads'].tolist()
                t_list = [t1_base] + best_pts['Tempo_Tratado'].tolist()
                max_y_local = max(max_y_local, max(t_list))

                cor = CORES_ESTRATEGIAS[est]
                mkr = MARCADORES[est]
                t4_val = t_list[-1]
                e_weak_4 = (t1_base / t4_val) * 100.0

                ax.plot(p_list, t_list, color=cor, linestyle='-', linewidth=2.5, alpha=0.9, zorder=4,
                        label=f'{est} (T4={t4_val:.1f}s, E={e_weak_4:.1f}%)')
                ax.scatter(p_list[1:], t_list[1:], color=cor, s=140, marker=mkr, edgecolor='#0f172a', linewidth=1.1, zorder=6)

            ax.axvline(2.0, color='#0284c7', linestyle='--', linewidth=1.1, alpha=0.6, zorder=2)
            ax.axvline(4.0, color='#64748b', linestyle=':', linewidth=1.1, alpha=0.6, zorder=2)

            ax.set_title(f'{cenario_nome} - {salto_nome}', fontsize=11.5, fontweight='bold', pad=10)
            ax.set_xlabel('Unidades de Processamento (p)', fontsize=10, fontweight='bold')
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(['1 (T1 Base)', '2 (Fís.)', '3 (SMT)', '4 (Máx.)'], fontsize=9)
            ax.set_xlim(0.7, 4.6)
            ax.set_ylim(0, max_y_local * 1.15)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=7.8)

        row_axes[0].set_ylabel('Tempo de Execução (s)', fontsize=11, fontweight='bold')

    plt.suptitle('Painel Consolidado de Escalabilidade Fraca (Lei de Gustafson): Tempo de Execução vs Processadores',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    fig.savefig(os.path.join(dir_png, "mpi_gustafson_tempo_linhas_consolidado.png"), dpi=300)
    fig.savefig(os.path.join(dir_pdf, "mpi_gustafson_tempo_linhas_consolidado.pdf"))
    plt.close(fig)
    print(" Salvo: relatorios/png/mpi_gustafson_tempo_linhas_consolidado.png")


def plotar_comparativo_amdahl_vs_gustafson(df_amdahl, df_gustafson, dir_png, dir_pdf):
    """
    Gráfico 3: O Painel Comparativo Unificado 2x2.
    Confronta diretamente as premissas e resultados de Amdahl (Forte) vs Gustafson (Fraca).
      - (0, 0): Amdahl Speedup (Saturação assintótica)
      - (0, 1): Gustafson Speedup Escalado (Crescimento proporcional)
      - (1, 0): Amdahl Eficiência Forte (Decaimento com p)
      - (1, 1): Gustafson Eficiência Fraca (Preservação de alto rendimento)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    cenario_foco = "Visão Completa"
    sub_amd = df_amdahl[(df_amdahl["Cenário"] == cenario_foco) & (df_amdahl["Resolução"].str.contains("16K"))]
    sub_gus = df_gustafson[(df_gustafson["Cenário"] == cenario_foco) & (df_gustafson["Salto_Carga"].str.contains("16K"))]

    p_vals = np.linspace(1.0, 8.0, 100)
    estrategias = ['Estático Cíclico', 'Dinâmico Meia-Linha', 'Dinâmico Dedicado', 'Estático Bloco']

    # --- PAINEL (0, 0): AMDAHL SPEEDUP (FORTE) ---
    ax00 = axes[0, 0]
    ax00.plot(p_vals, p_vals, color=CORES_ESTRATEGIAS['Ideal Linear'], linestyle='--', linewidth=1.5, label='Ideal ($S_p = p$)')
    for est in estrategias:
        row = sub_amd[sub_amd["Estratégia"] == est]
        if not row.empty:
            s_kf = float(row["s_karp_flatt_clamped"].iloc[0])
            sp4 = float(row["Speedup"].iloc[0])
            sp_curv = 1.0 / (s_kf + (1.0 - s_kf) / p_vals)
            ax00.plot(p_vals, sp_curv, color=CORES_ESTRATEGIAS[est], label=f"{est} (s={s_kf*100:.1f}%)", linewidth=2.0)
            ax00.scatter([4.0], [sp4], color=CORES_ESTRATEGIAS[est], s=70, edgecolor='black', zorder=5)

    ax00.set_title('Amdahl (Escalabilidade Forte) - Speedup\n[Problema de Tamanho Fixo: 16K]', fontsize=11.5, fontweight='bold')
    ax00.set_xlabel('Processadores ($p$)', fontsize=10)
    ax00.set_ylabel('Speedup ($S_p$)', fontsize=10, fontweight='bold')
    ax00.set_xlim(0.8, 8.2)
    ax00.set_ylim(0, 5.0)
    ax00.legend(fontsize=8, loc='upper left')

    # --- PAINEL (0, 1): GUSTAFSON SPEEDUP (FRACA) ---
    ax01 = axes[0, 1]
    ax01.plot(p_vals, p_vals, color=CORES_ESTRATEGIAS['Ideal Linear'], linestyle='--', linewidth=1.5, label='Ideal Gustafson ($S = p$)')
    for est in estrategias:
        row_g = sub_gus[sub_gus["Estratégia"] == est]
        if not row_g.empty:
            s_g = float(row_g["Fracao_Serial_Gustafson"].iloc[0])
            sp_scaled = float(row_g["Speedup_Escalado"].iloc[0])
            # Reta de Gustafson: S(p) = p - s*(p - 1)
            sp_gust_curv = p_vals - s_g * (p_vals - 1.0)
            ax01.plot(p_vals, sp_gust_curv, color=CORES_ESTRATEGIAS[est], label=f"{est} ($S_4={sp_scaled:.2f}\\times$)", linewidth=2.0)
            ax01.scatter([4.0], [sp_scaled], color=CORES_ESTRATEGIAS[est], s=70, edgecolor='black', zorder=5)

    ax01.set_title('Gustafson (Escalabilidade Fraca) - Speedup Escalado\n[Carga Cresce Proporcionalmente: 8K → 16K]', fontsize=11.5, fontweight='bold')
    ax01.set_xlabel('Processadores ($p$)', fontsize=10)
    ax01.set_ylabel('Speedup Escalado ($S_{scaled}$)', fontsize=10, fontweight='bold')
    ax01.set_xlim(0.8, 8.2)
    ax01.set_ylim(0, 5.0)
    ax01.legend(fontsize=8, loc='upper left')

    # --- PAINEL (1, 0): AMDAHL EFICIÊNCIA (FORTE) ---
    ax10 = axes[1, 0]
    ax10.axhline(100.0, color=CORES_ESTRATEGIAS['Ideal Linear'], linestyle='--', linewidth=1.5, label='Ideal 100%')
    ax10.axhline(50.0, color='#94a3b8', linestyle=':', linewidth=1.2, label='Patamar 50%')
    for est in estrategias:
        row = sub_amd[sub_amd["Estratégia"] == est]
        if not row.empty:
            s_kf = float(row["s_karp_flatt_clamped"].iloc[0])
            ef_curv = (1.0 / (s_kf * p_vals + (1.0 - s_kf))) * 100.0
            ax10.plot(p_vals, ef_curv, color=CORES_ESTRATEGIAS[est], label=est, linewidth=2.0)
            ef4 = float(row["Ep=4_%"].iloc[0])
            ax10.scatter([4.0], [ef4], color=CORES_ESTRATEGIAS[est], s=70, edgecolor='black', zorder=5)

    ax10.set_title('Amdahl (Escalabilidade Forte) - Eficiência\n[Queda acentuada conforme p aumenta]', fontsize=11.5, fontweight='bold')
    ax10.set_xlabel('Processadores ($p$)', fontsize=10)
    ax10.set_ylabel('Eficiência Forte ($E_p$ %)', fontsize=10, fontweight='bold')
    ax10.set_xlim(0.8, 8.2)
    ax10.set_ylim(0, 110)
    ax10.legend(fontsize=8, loc='upper right')

    # --- PAINEL (1, 1): GUSTAFSON EFICIÊNCIA (FRACA) ---
    ax11 = axes[1, 1]
    ax11.axhline(100.0, color=CORES_ESTRATEGIAS['Ideal Linear'], linestyle='--', linewidth=1.5, label='Ideal 100% (Tempo Constante)')
    ax11.axhline(50.0, color='#94a3b8', linestyle=':', linewidth=1.2, label='Patamar 50%')
    for est in estrategias:
        row_g = sub_gus[sub_gus["Estratégia"] == est]
        if not row_g.empty:
            s_g = float(row_g["Fracao_Serial_Gustafson"].iloc[0])
            # E_gust(p) = S_gust(p) / p = 1 - s*(1 - 1/p)
            ef_gust_curv = (1.0 - s_g * (1.0 - 1.0 / p_vals)) * 100.0
            ax11.plot(p_vals, ef_gust_curv, color=CORES_ESTRATEGIAS[est], label=est, linewidth=2.0)
            ef_real = float(row_g["Eficiencia_Fraca_%"].iloc[0])
            ax11.scatter([4.0], [ef_real], color=CORES_ESTRATEGIAS[est], s=70, edgecolor='black', zorder=5)

    ax11.set_title('Gustafson (Escalabilidade Fraca) - Eficiência Fraca\n[Alta preservação: carga cresce com p]', fontsize=11.5, fontweight='bold')
    ax11.set_xlabel('Processadores ($p$)', fontsize=10)
    ax11.set_ylabel('Eficiência Fraca ($E_{weak}$ %)', fontsize=10, fontweight='bold')
    ax11.set_xlim(0.8, 8.2)
    ax11.set_ylim(0, 110)
    ax11.legend(fontsize=8, loc='lower left')

    plt.suptitle('Confronto Teórico-Experimental: Lei de Amdahl (Carga Fixa) vs. Lei de Gustafson (Carga Escalonada)',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    fig.savefig(os.path.join(dir_png, "mpi_comparativo_amdahl_gustafson.png"), dpi=300)
    fig.savefig(os.path.join(dir_pdf, "mpi_comparativo_amdahl_gustafson.pdf"))
    plt.close(fig)
    print(" Salvo: relatorios/png/mpi_comparativo_amdahl_gustafson.png")


def plotar_escalabilidade_resolucoes(df_best, dir_png, dir_pdf):
    """
    Gráfico 4: Evolução de Amdahl conforme a Resolução Aumenta (4K -> 8K -> 16K).
    Demonstra que resoluções maiores reduzem a fração serial relativa (overhead de rede
    fica menor frente ao trabalho de cálculo), elevando o speedup.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]
    resolucoes = ["4096 × 4096 (4K)", "8192 × 8192 (8K)", "16384 × 16384 (16K)"]
    labels_res = ["4K", "8K", "16K"]
    estrategias = ['Estático Cíclico', 'Dinâmico Meia-Linha', 'Dinâmico Dedicado', 'Estático Bloco', 'Dinâmico Participativo']

    for cenario_nome, ax in cenarios:
        sub_c = df_best[df_best["Cenário"] == cenario_nome]
        pontos_para_anotar = []

        for est in estrategias:
            sp_vals = []
            for r in resolucoes:
                sub_r = sub_c[(sub_c["Resolução"] == r) & (sub_c["Estratégia"] == est)]
                sp_vals.append(sub_r["Speedup"].iloc[0] if not sub_r.empty else np.nan)

            ax.plot(labels_res, sp_vals, marker=MARCADORES[est], color=CORES_ESTRATEGIAS[est],
                    linewidth=2.2, markersize=8, label=est)
            for x_idx, sp in enumerate(sp_vals):
                if pd.notna(sp):
                    pontos_para_anotar.append((x_idx, sp, f"{sp:.2f}×", CORES_ESTRATEGIAS[est]))

        annotar_pontos_sem_colisao(ax, pontos_para_anotar, x_is_category=True)

        ax.axhline(4.0, color=CORES_ESTRATEGIAS['Ideal Linear'], linestyle='--', linewidth=1.5, label='Ideal 4 Cores (4×)')
        ax.axhline(2.0, color='#64748b', linestyle=':', linewidth=1.3, label='2 Cores Físicos (2×)')
        ax.set_title(f'Cenário: {cenario_nome}', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Resolução do Problema', fontsize=11, fontweight='bold')
        ax.set_ylim(0.0, 4.4)

    axes[0].set_ylabel('Speedup Obtido ($S_4 = T_1 / T_4$)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=8.5)
    axes[0].legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=8.5)

    plt.suptitle('Impacto da Resolução sobre a Escalabilidade Forte (4K → 8K → 16K sob p=4 Processos)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    fig.savefig(os.path.join(dir_png, "mpi_escalabilidade_resolucoes_4k_8k_16k.png"), dpi=300)
    fig.savefig(os.path.join(dir_pdf, "mpi_escalabilidade_resolucoes_4k_8k_16k.pdf"))
    plt.close(fig)
    print(" Salvo: relatorios/png/mpi_escalabilidade_resolucoes_4k_8k_16k.png")


def plotar_escalabilidade_empirica_512(df_512, dir_png, dir_pdf):
    """
    Gráfico 5: Escalabilidade Empírica Discreta com p = 1, 2, 3, 4 (Resolução 512x512).
    Apresenta a curva empírica exata para cada número de processos medido em laboratório.
    """
    if df_512.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]

    def map_strat_name(code):
        if code == "mpi_estatico": return "Estático Cíclico"
        if code == "mpi_dinamico": return "Dinâmico Dedicado"
        if code == "mpi_dinamico_meia_linha": return "Dinâmico Meia-Linha"
        if code == "mpi_dinamico_participativo": return "Dinâmico Participativo"
        return code

    ponto_times = df_512[df_512["Code"] == "ponto_a_ponto"].groupby("Cenário")["TempoGasto"].mean().to_dict()

    for cenario_nome, ax in cenarios:
        sub_c = df_512[df_512["Cenário"] == cenario_nome]
        t1 = ponto_times.get(cenario_nome, 0.20)

        # Ideal
        ax.plot([1, 2, 3, 4], [1, 2, 3, 4], color=CORES_ESTRATEGIAS['Ideal Linear'], linestyle='--',
                linewidth=1.5, label='Ideal ($S_p = p$)')

        pontos_para_anotar = []

        for code in ["mpi_estatico", "mpi_dinamico_meia_linha", "mpi_dinamico", "mpi_dinamico_participativo"]:
            sub_code = sub_c[sub_c["Code"] == code]
            if sub_code.empty:
                continue

            grp = sub_code.groupby("Threads")["TempoGasto"].mean().reset_index()
            # Adiciona p=1 como 1.0 (baseline)
            threads_list = [1] + grp["Threads"].tolist()
            speedups_list = [1.0] + [t1 / t for t in grp["TempoGasto"].tolist()]

            est_nome = map_strat_name(code)
            cor_est = CORES_ESTRATEGIAS.get(est_nome, '#64748b')
            ax.plot(threads_list, speedups_list, marker=MARCADORES.get(est_nome, 'o'),
                    color=cor_est, linewidth=2.0, label=est_nome)

            for p_val, sp_val in zip(threads_list[1:], speedups_list[1:]):
                pontos_para_anotar.append((p_val, sp_val, f"{sp_val:.2f}×", cor_est))

        annotar_pontos_sem_colisao(ax, pontos_para_anotar, x_is_category=False)

        ax.set_title(f'Cenário: {cenario_nome} (Matriz 512×512)', fontsize=12.5, fontweight='bold', pad=12)
        ax.set_xlabel('Número de Processos ($p$)', fontsize=11, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(0, 4.3)

    axes[0].set_ylabel('Speedup Empírico ($T_1 / T_p$)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=8.5)
    axes[0].legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=8.5)

    plt.suptitle('Curvas de Escalabilidade Empírica com Variação Discreta de Processos ($p=1, 2, 3, 4$)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    fig.savefig(os.path.join(dir_png, "mpi_escalabilidade_forte_512_p1_a_p4.png"), dpi=300)
    fig.savefig(os.path.join(dir_pdf, "mpi_escalabilidade_forte_512_p1_a_p4.pdf"))
    plt.close(fig)
    print(" Salvo: relatorios/png/mpi_escalabilidade_forte_512_p1_a_p4.png")


# ==============================================================================
# EXPORTAÇÃO CSV CONSOLIDADO DE ESCALABILIDADE
# ==============================================================================
def exportar_csv_escalabilidade(df_amdahl, df_gustafson, out_csv_path):
    """Gera o arquivo CSV analítico com todas as métricas de Amdahl e Gustafson."""
    amd_records = []
    for _, row in df_amdahl.iterrows():
        amd_records.append({
            "Regime": "Escalabilidade Forte (Amdahl)",
            "Cenário": row["Cenário"],
            "Estratégia": row["Estratégia"],
            "Configuração_Carga": f"Tamanho Fixo: {row['Resolução']}",
            "Processadores_p": 4,
            "Tempo_T1_s": row["T1_s"],
            "Tempo_Tp_s": row["Tempo_s"],
            "Speedup_Experimental": row["Speedup"],
            "Eficiencia_Experimental_%": row["Ep=4_%"],
            "Fracao_Serial_s": row["s_karp_flatt_clamped"],
            "Fracao_Paralela_f": row["f_paralela"],
            "Limite_Assintotico_Smax": row["S_max_amdahl"],
            "Eficiencia_Gustafson_%": np.nan,
            "Speedup_Escalado_Gustafson": np.nan
        })

    gus_records = []
    for _, row in df_gustafson.iterrows():
        gus_records.append({
            "Regime": "Escalabilidade Fraca (Gustafson)",
            "Cenário": row["Cenário"],
            "Estratégia": row["Estratégia"],
            "Configuração_Carga": f"Expansão: {row['Salto_Carga']}",
            "Processadores_p": 4,
            "Tempo_T1_s": row["Tempo_T1_Base_s"],
            "Tempo_Tp_s": row["Tempo_Tp_Alvo_s"],
            "Speedup_Experimental": row["Speedup_Escalado"],
            "Eficiencia_Experimental_%": row["Eficiencia_Fraca_%"],
            "Fracao_Serial_s": row["Fracao_Serial_Gustafson"],
            "Fracao_Paralela_f": 1.0 - row["Fracao_Serial_Gustafson"],
            "Limite_Assintotico_Smax": np.nan,  # Gustafson não tem teto 1/s
            "Eficiencia_Gustafson_%": row["Eficiencia_Fraca_%"],
            "Speedup_Escalado_Gustafson": row["Speedup_Escalado"]
        })

    df_consolidado = pd.DataFrame(amd_records + gus_records)
    df_consolidado.to_csv(out_csv_path, index=False)
    print(f"✅ Tabela consolidada de escalabilidade salva em: '{out_csv_path}'")
    return df_consolidado


# ==============================================================================
# RELATÓRIO FORMATADO NO TERMINAL
# ==============================================================================
def imprimir_relatorio_terminal(df_amdahl, df_gustafson):
    """Exibe no terminal a síntese teórica e empírica das leis de escalabilidade."""
    sep = "=" * 105
    print("\n" + sep)
    print(" 🚀 RELATÓRIO CONSOLIDADO DE ESCALABILIDADE: LEIS DE AMDAHL E GUSTAFSON-BARSIS")
    print(sep)

    print("\n📌 1. ESCALABILIDADE FORTE (STRONG SCALING - LEI DE AMDAHL)")
    print("   Premissa: Tamanho global estritamente fixo (16384 × 16384). Variação de p.")
    print("   Modelo  : S(p) = 1 / (s + (1-s)/p) | Fração serial s estimada via Karp-Flatt")
    print("   Limite  : S_max = 1 / s (quando p -> infinito)")
    print("-" * 105)

    sub_16k = df_amdahl[df_amdahl["Resolução"].str.contains("16K")].sort_values(by=["Cenário", "Speedup"], ascending=[True, False])
    cols_amd = ["Cenário", "Estratégia", "T1_s", "Tempo_s", "Speedup", "Ep=4_%", "s_karp_flatt_clamped", "f_paralela", "S_max_amdahl"]
    display_amd = sub_16k[cols_amd].copy()
    display_amd.columns = ["Cenário", "Estratégia", "T1 (s)", "Tp (s)", "Sp (p=4)", "Ep (%)", "Fração s", "Fração f", "Limite S_max"]
    print(display_amd.to_string(index=False))

    print("\n" + "-" * 105)
    print("📌 2. ESCALABILIDADE FRACA (WEAK SCALING - LEI DE GUSTAFSON-BARSIS)")
    print("   Premissa: A carga total expande-se proporcionalmente a p (carga/proc = constante).")
    print("   Modelo  : E_weak = T1(W1) / Tp(p*W1) | S_scaled = p * E_weak | S_Gustafson = p - s*(p-1)")
    print("-" * 105)

    cols_gus = ["Cenário", "Estratégia", "Salto_Carga", "Tempo_T1_Base_s", "Tempo_Tp_Alvo_s", "Eficiencia_Fraca_%", "Speedup_Escalado", "Eficiencia_Fisica_Fraca_%"]
    display_gus = df_gustafson[cols_gus].copy()
    display_gus.columns = ["Cenário", "Estratégia", "Expansão de Carga", "T1 Base", "Tp Alvo", "E_weak (%)", "S_scaled", "E_físico (%)"]
    print(display_gus.to_string(index=False))

    print("\n" + sep)
    print("💡 PRINCIPAIS CONCLUSÕES E INSIGHTS TÉCNICOS:")
    print(" 1. [Amdahl e o Gargalo Serial]:")
    print("    - A estratégia 'Estático Cíclico' alcança a maior fração paralela (f ≈ 72.4% em Visão Completa e 64.2% em Zoom).")
    print("    - O limite assintótico teórico de Amdahl (S_max) para o Cíclico é de ~3.63x, demonstrando excelente paralelismo.")
    print("    - 'Dinâmico Dedicado' tem fração paralela limitada a ~41%, pois 1 dos 4 processos atua estritamente como coordenador.")
    print(" 2. [Gustafson e a Preservação do Rendimento]:")
    print("    - No regime fraco, ao expandir a resolução de 4K (p=1) para 8K (p=4) e 16K (p=4), a eficiência do 'Estático Cíclico'")
    print("      permanece alta (51.4% a 54.8%), superando 100% quando normalizada sobre os 2 núcleos físicos reais.")
    print("    - O 'Dinâmico Meia-Linha' demonstra expressivo ganho no cenário Zoom 16K (Eficiência Fraca de 43.0% e S_scaled = 1.72x),")
    print("      comprovando que granularidade fina mitiga o desbalanceamento severo em regiões de alta densidade iterativa.")
    print(sep + "\n")


# ==============================================================================
# PONTO DE ENTRADA CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Geração de Gráficos de Escalabilidade (Amdahl & Gustafson)")
    parser.add_argument("--dir-png", default=resolve_path("relatorios/png"), help="Diretório de saída para gráficos PNG")
    parser.add_argument("--dir-pdf", default=resolve_path("relatorios/pdf"), help="Diretório de saída para gráficos PDF")
    parser.add_argument("--out-csv", default=resolve_path("data/analise_amdal_gustafson.csv"), help="Caminho do CSV consolidado")
    args = parser.parse_args()

    os.makedirs(args.dir_png, exist_ok=True)
    os.makedirs(args.dir_pdf, exist_ok=True)

    configurar_estilo()
    df_best, t1_map, df_512, _ = carregar_dados()

    df_amdahl = processar_escalabilidade_forte(df_best, p_max=4)
    df_gustafson = processar_escalabilidade_fraca(df_best, t1_map)

    # Exporta CSV
    exportar_csv_escalabilidade(df_amdahl, df_gustafson, args.out_csv)

    # Renderiza os gráficos de escalabilidade Forte (Amdahl)
    plotar_escalabilidade_forte_amdahl(df_amdahl, args.dir_png, args.dir_pdf)
    plotar_escalabilidade_forte_amdahl_4k(args.dir_png, args.dir_pdf)
    plotar_escalabilidade_forte_amdahl_8k(args.dir_png, args.dir_pdf)
    plotar_escalabilidade_forte_amdahl_16k(args.dir_png, args.dir_pdf)

    # Renderiza os gráficos de escalabilidade Fraca (Gustafson) - FORMA 1 (Tempo vs Processadores)
    plotar_gustafson_tempo_salto1(args.dir_png, args.dir_pdf)
    plotar_gustafson_tempo_salto2(args.dir_png, args.dir_pdf)
    plotar_gustafson_tempo_consolidado(args.dir_png, args.dir_pdf)

    # Renderiza os gráficos de escalabilidade Fraca (Gustafson) - FORMA 2 (Eficiência e Speedup em Linhas)
    plotar_escalabilidade_fraca_gustafson(df_gustafson, args.dir_png, args.dir_pdf)
    plotar_gustafson_speedup_escalado_linhas(df_gustafson, args.dir_png, args.dir_pdf)

    # Gráficos complementares
    plotar_comparativo_amdahl_vs_gustafson(df_amdahl, df_gustafson, args.dir_png, args.dir_pdf)
    plotar_escalabilidade_resolucoes(df_best, args.dir_png, args.dir_pdf)
    plotar_escalabilidade_empirica_512(df_512, args.dir_png, args.dir_pdf)

    # Exibe resumo no terminal
    imprimir_relatorio_terminal(df_amdahl, df_gustafson)


if __name__ == "__main__":
    main()
