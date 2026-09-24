#!/usr/bin/env python3
"""
Geração e Exportação de Gráficos de Desempenho Paralelo (Mandelbrot OpenMP)
Exporta os 18 gráficos separados tanto em PNG (300 DPI) quanto em PDF Vetorial,
além de gerar um relatório PDF consolidado multi-páginas (relatorio_graficos_mandelbrot.pdf).
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from metricas import load_dataset, calculate_metrics

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Estilo visual moderno e limpo
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

CORES_SCHEDULE = {
    'static': '#3b82f6',     # Azul
    'dynamic': '#10b981',    # Verde Esmeralda
    'guided': '#8b5cf6',     # Roxo
    'auto': '#f59e0b',       # Âmbar
    'sequencial': '#64748b'  # Cinza Ardósia
}

CENARIOS_LISTA = [
    {
        'name': 'Visão Completa',
        'width': 4096,
        'res_label': '4096×4096 (4K - 16.7M px)',
        'tag': 'visao_completa_4k',
        'desc': 'Visão Completa - 4096×4096 (Max Iter: 1000)'
    },
    {
        'name': 'Visão Completa',
        'width': 8192,
        'res_label': '8192×8192 (8K - 67.1M px)',
        'tag': 'visao_completa_8k',
        'desc': 'Visão Completa - 8192×8192 (Max Iter: 1000)'
    },
    {
        'name': 'Visão Completa',
        'width': 16384,
        'res_label': '16384×16384 (16K - 268.4M px)',
        'tag': 'visao_completa_16k',
        'desc': 'Visão Completa - 16384×16384 (Max Iter: 1000)'
    },
    {
        'name': 'Zoom (Seahorse)',
        'width': 4096,
        'res_label': '4096×4096 (4K - 16.7M px)',
        'tag': 'zoom_seahorse_4k',
        'desc': 'Zoom (Seahorse) - 4096×4096 (Max Iter: 5000)'
    },
    {
        'name': 'Zoom (Seahorse)',
        'width': 8192,
        'res_label': '8192×8192 (8K - 67.1M px)',
        'tag': 'zoom_seahorse_8k',
        'desc': 'Zoom (Seahorse) - 8192×8192 (Max Iter: 5000)'
    },
    {
        'name': 'Zoom (Seahorse)',
        'width': 16384,
        'res_label': '16384×16384 (16K - 268.4M px)',
        'tag': 'zoom_seahorse_16k',
        'desc': 'Zoom (Seahorse) - 16384×16384 (Max Iter: 5000)'
    }
]


def carregar_dados(csv_path=None, machine="deCasa", etapa=None):
    """Carrega dataset e calcula métricas isolando 1 e 4 threads."""
    df_raw = load_dataset(csv_path, etapa=etapa)
    if machine and machine.lower() != 'all':
        df_raw = df_raw[df_raw['Machine'] == machine].copy()
    
    # Filtra estritamente execuções com 1 e 4 threads
    df_raw = df_raw[df_raw['Threads'].isin([1, 4])].copy()
    df_metrics = calculate_metrics(df_raw)
    return df_metrics


def salvar_figura(fig, nome_base, dir_png, dir_pdf, pdf_pages=None):
    """Salva a figura em formato PNG de alta resolução e em formato PDF Vetorial."""
    path_png = os.path.join(dir_png, f"{nome_base}.png")
    path_pdf = os.path.join(dir_pdf, f"{nome_base}.pdf")
    
    # 1. Salva PNG (Raster 300 DPI)
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    
    # 2. Salva PDF Individual (Vetorial)
    fig.savefig(path_pdf, format='pdf', bbox_inches='tight')
    
    # 3. Adiciona ao relatório PDF multipáginas (se fornecido)
    if pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches='tight')
        
    plt.close(fig)
    print(f"  📄 PNG: {path_png}  |  📑 PDF: {path_pdf}")


# ==============================================================================
# TIPO 1: DESEMPENHO POR POLÍTICA DE ESCALONAMENTO (SCHEDULE & CHUNK)
# ==============================================================================
def gerar_tipo1_schedules(df, cenario_info, dir_png, dir_pdf, pdf_pages=None):
    """Gera gráfico individual de barras horizontais comparando os escalonamentos."""
    sc = cenario_info['name']
    w = cenario_info['width']
    tag = cenario_info['tag']
    desc = cenario_info['desc']

    sub = df[(df['Scenario'] == sc) & (df['WIDTH'] == w) & (df['Threads'] == 4) & (df['Code'] == 'paralelo_collapse')].copy()
    if sub.empty:
        sub = df[(df['Scenario'] == sc) & (df['WIDTH'] == w) & (df['Threads'] == 4) & (df['Code'] == 'paralelo')].copy()

    if sub.empty:
        return

    sub['Config'] = sub['Schedule'].str.capitalize() + " (chunk=" + sub['ChunkSize'].astype(str) + ")"

    agg = sub.groupby(['Schedule', 'Config']).agg(
        Tempo_Medio=('TempoGasto', 'mean'),
        Speedup_Medio=('Speedup', 'mean')
    ).reset_index().sort_values('Tempo_Medio', ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = [CORES_SCHEDULE.get(s, '#3b82f6') for s in agg['Schedule']]
    bars = ax.barh(agg['Config'], agg['Tempo_Medio'], color=colors, height=0.62, edgecolor='#1e293b', alpha=0.9)

    max_val = max(agg['Tempo_Medio'])
    for bar, speedup in zip(bars, agg['Speedup_Medio']):
        width = bar.get_width()
        ax.text(width + (max_val * 0.02), bar.get_y() + bar.get_height()/2,
                f"{width:.2f}s  ({speedup:.2f}x)",
                ha='left', va='center', fontsize=9.5, fontweight='bold', color='#1e293b')

    ax.set_title(f'Desempenho por Escalonamento OpenMP (4 Threads)\n{desc}', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Tempo Médio de Execução (segundos) - Menor é melhor', fontsize=10.5)
    ax.set_xlim(0, max_val * 1.38)
    ax.grid(axis='x', alpha=0.6)

    # Legenda na parte superior direita
    legend_elements = [
        Patch(facecolor=CORES_SCHEDULE['dynamic'], label='Dynamic (Balanceamento Dinâmico)'),
        Patch(facecolor=CORES_SCHEDULE['guided'], label='Guided (Granularidade Decrescente)'),
        Patch(facecolor=CORES_SCHEDULE['static'], label='Static (Divisão Estática)'),
        Patch(facecolor=CORES_SCHEDULE['auto'], label='Auto (OpenMP Default)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8.5, framealpha=0.95)
    plt.tight_layout()

    salvar_figura(fig, f"tipo1_schedules_{tag}", dir_png, dir_pdf, pdf_pages)


# ==============================================================================
# TIPO 2: COMPARATIVO PARALELO 1D VS PARALELO 2D (COLLAPSE)
# ==============================================================================
def gerar_tipo2_collapse_vs_1d(df, cenario_info, dir_png, dir_pdf, pdf_pages=None):
    """Gera gráfico individual de barras comparando Paralelo 1D vs Collapse 2D."""
    sc = cenario_info['name']
    w = cenario_info['width']
    tag = cenario_info['tag']
    desc = cenario_info['desc']

    sub = df[(df['Scenario'] == sc) & (df['WIDTH'] == w) & (df['Threads'] == 4) & (df['Code'].isin(['paralelo', 'paralelo_collapse']))].copy()
    if sub.empty:
        return

    sub['Config'] = sub['Schedule'].str.capitalize() + " (" + sub['ChunkSize'].astype(str) + ")"

    piv = sub.pivot_table(index='Config', columns='Code', values='TempoGasto', aggfunc='mean').dropna().reset_index()
    if 'paralelo' not in piv.columns or 'paralelo_collapse' not in piv.columns:
        return

    piv['Ganho_%'] = ((piv['paralelo'] - piv['paralelo_collapse']) / piv['paralelo']) * 100.0
    piv = piv.sort_values('paralelo', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = np.arange(len(piv))
    height = 0.38

    bars1 = ax.barh(y_pos + height/2, piv['paralelo'], height, label='Paralelo 1D (Loop Externo)', color='#3b82f6', edgecolor='#1e293b', alpha=0.9)
    bars2 = ax.barh(y_pos - height/2, piv['paralelo_collapse'], height, label='Paralelo 2D (Collapse OpenMP)', color='#10b981', edgecolor='#1e293b', alpha=0.9)

    max_val = max(piv[['paralelo', 'paralelo_collapse']].max())
    for y, (t1d, tcol, ganho) in enumerate(zip(piv['paralelo'], piv['paralelo_collapse'], piv['Ganho_%'])):
        sinal = "+" if ganho >= 0 else ""
        cor_txt = "#047857" if ganho >= 0 else "#b91c1c"
        ax.text(max(t1d, tcol) + (max_val * 0.02), y, f"{sinal}{ganho:.1f}%",
                va='center', fontsize=9, fontweight='bold', color=cor_txt)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(piv['Config'], fontsize=9.5)
    ax.set_title(f'Comparativo: Paralelo 1D vs Paralelo Collapse 2D (4 Threads)\n{desc}', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Tempo Médio de Execução (segundos) - Menor é melhor', fontsize=10.5)
    ax.set_xlim(0, max_val * 1.25)
    ax.grid(axis='x', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9.5)
    plt.tight_layout()

    salvar_figura(fig, f"tipo2_collapse_vs_1d_{tag}", dir_png, dir_pdf, pdf_pages)


# ==============================================================================
# TIPO 3: MÉTRICAS DE DESEMPENHO (TEMPO, SPEEDUP E EFICIÊNCIA: 1 VS 4 THREADS)
# ==============================================================================
def gerar_tipo3_metricas(df, cenario_info, dir_png, dir_pdf, pdf_pages=None):
    """Gera painel com 3 subplots mostrando Tempo, Speedup e Eficiência (1 vs 4 Threads)."""
    sc = cenario_info['name']
    w = cenario_info['width']
    tag = cenario_info['tag']
    desc = cenario_info['desc']

    sub_all = df[(df['Scenario'] == sc) & (df['WIDTH'] == w)].copy()
    if sub_all.empty:
        return

    # Baseline T1: utiliza estritamente ponto_a_ponto (carga completa de trabalho 1-thread)
    pap_sub = sub_all[sub_all['Code'] == 'ponto_a_ponto']
    if not pap_sub.empty:
        t_seq = pap_sub['TempoGasto'].mean()
    elif 'T1' in sub_all.columns and not pd.isna(sub_all['T1'].iloc[0]):
        t_seq = sub_all['T1'].iloc[0]
    else:
        t_seq = sub_all[sub_all['Code'] == 'sequencial']['TempoGasto'].mean()

    estrategias = [
        {'code': 'paralelo_collapse', 'sched': 'dynamic', 'chunk': 1024, 'label': 'Collapse - Dynamic (1024)', 'color': '#10b981', 'marker': 'o'},
        {'code': 'paralelo_collapse', 'sched': 'guided', 'chunk': 1024, 'label': 'Collapse - Guided (1024)', 'color': '#8b5cf6', 'marker': 'v'},
        {'code': 'paralelo', 'sched': 'dynamic', 'chunk': 64, 'label': '1D - Dynamic (64)', 'color': '#06b6d4', 'marker': '^'},
        {'code': 'paralelo', 'sched': 'static', 'chunk': 64, 'label': '1D - Static (64)', 'color': '#ef4444', 'marker': 's'},
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Métricas de Desempenho (1 vs 4 Threads)\n{desc}', fontsize=13, fontweight='bold', y=1.03)

    thread_range = [1, 4]
    ax2.plot(thread_range, thread_range, '--', color='#94a3b8', label='Ideal ($S_p = p$)', linewidth=1.5)
    ax3.axhline(100, linestyle='--', color='#94a3b8', label='Ideal (100%)', linewidth=1.5)

    for est in estrategias:
        sub_est = sub_all[(sub_all['Code'] == est['code']) & (sub_all['Schedule'] == est['sched']) & (sub_all['ChunkSize'] == est['chunk']) & (sub_all['Threads'] == 4)]
        if sub_est.empty:
            continue
        
        t4 = sub_est['TempoGasto'].mean()
        s4 = t_seq / t4
        e4 = (s4 / 4.0) * 100.0

        x_vals = [1, 4]
        y_tempos = [t_seq, t4]
        y_speedups = [1.0, s4]
        y_eficiencias = [100.0, e4]

        # Plot 1: Tempo
        ax1.plot(x_vals, y_tempos, marker=est['marker'], color=est['color'], label=est['label'], linewidth=2.2, markersize=8)
        ax1.annotate(f"{t4:.2f}s", (4, t4), textcoords="offset points", xytext=(8, -2), fontsize=8.5, fontweight='bold', color=est['color'])

        # Plot 2: Speedup
        ax2.plot(x_vals, y_speedups, marker=est['marker'], color=est['color'], label=est['label'], linewidth=2.2, markersize=8)
        ax2.annotate(f"{s4:.2f}x", (4, s4), textcoords="offset points", xytext=(8, -2), fontsize=8.5, fontweight='bold', color=est['color'])

        # Plot 3: Eficiência
        ax3.plot(x_vals, y_eficiencias, marker=est['marker'], color=est['color'], label=est['label'], linewidth=2.2, markersize=8)
        ax3.annotate(f"{e4:.1f}%", (4, e4), textcoords="offset points", xytext=(8, -2), fontsize=8.5, fontweight='bold', color=est['color'])

    # Configuração Eixos
    ax1.set_title('Tempo de Execução (s)', fontsize=11.5, fontweight='bold')
    ax1.set_xlabel('Threads', fontsize=10.5)
    ax1.set_ylabel('Tempo (segundos)', fontsize=10.5)
    ax1.set_xticks(thread_range)
    ax1.set_xticklabels(['1 Thread\n(Sequencial)', '4 Threads\n(OpenMP)'])
    ax1.set_xlim(0.7, 4.65)
    ax1.grid(True)
    ax1.legend(loc='upper right', fontsize=8.5)

    ax2.set_title('Speedup Relativo ($S_p$)', fontsize=11.5, fontweight='bold')
    ax2.set_xlabel('Threads', fontsize=10.5)
    ax2.set_ylabel('Speedup ($S_p = T_1 / T_p$)', fontsize=10.5)
    ax2.set_xticks(thread_range)
    ax2.set_xticklabels(['1 Thread\n(Base)', '4 Threads\n(OpenMP)'])
    ax2.set_xlim(0.7, 4.65)
    ax2.set_ylim(0.5, 4.85)
    ax2.grid(True)
    ax2.legend(loc='upper left', fontsize=8.5)

    ax3.set_title('Eficiência de Processamento ($E_p$)', fontsize=11.5, fontweight='bold')
    ax3.set_xlabel('Threads', fontsize=10.5)
    ax3.set_ylabel('Eficiência (%)', fontsize=10.5)
    ax3.set_xticks(thread_range)
    ax3.set_xticklabels(['1 Thread\n(100%)', '4 Threads\n(OpenMP)'])
    ax3.set_xlim(0.7, 4.65)
    ax3.set_ylim(30, 120)
    ax3.grid(True)
    ax3.legend(loc='lower left', fontsize=8.5)
    plt.tight_layout()

    salvar_figura(fig, f"tipo3_metricas_{tag}", dir_png, dir_pdf, pdf_pages)


# ==============================================================================
# TIPO 4: COMPARAÇÃO DE ESCALONAMENTOS AO LONGO DAS RESOLUÇÕES (4K, 8K, 16K)
# ==============================================================================
def gerar_tipo4_schedules_multi_res(df, sc_name, tag, title_desc, dir_png, dir_pdf, pdf_pages=None):
    """
    Gera painel comparativo 1x3 com as políticas de escalonamento lado a lado
    ao longo das três resoluções (4096, 8192, 16384) para um determinado cenário.
    """
    sub_sc = df[(df['Scenario'] == sc_name) & (df['Threads'] == 4) & (df['Code'] == 'paralelo_collapse')].copy()
    if sub_sc.empty:
        sub_sc = df[(df['Scenario'] == sc_name) & (df['Threads'] == 4) & (df['Code'] == 'paralelo')].copy()
    if sub_sc.empty:
        return

    sub_sc['Config'] = sub_sc['Schedule'].str.capitalize() + " (chunk=" + sub_sc['ChunkSize'].astype(str) + ")"

    widths = [4096, 8192, 16384]
    res_titles = ['4096×4096 (4K)', '8192×8192 (8K)', '16384×16384 (16K)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
    fig.suptitle(f'Comparação das Políticas de Escalonamento ao Longo das Resoluções (4 Threads)\n{title_desc}',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, w, res_t in zip(axes, widths, res_titles):
        sub_w = sub_sc[sub_sc['WIDTH'] == w].copy()
        if sub_w.empty:
            continue

        agg = sub_w.groupby(['Schedule', 'Config']).agg(
            Tempo_Medio=('TempoGasto', 'mean'),
            Speedup_Medio=('Speedup', 'mean')
        ).reset_index().sort_values('Tempo_Medio', ascending=False)

        colors = [CORES_SCHEDULE.get(s, '#3b82f6') for s in agg['Schedule']]
        bars = ax.barh(agg['Config'], agg['Tempo_Medio'], color=colors, height=0.62, edgecolor='#1e293b', alpha=0.9)

        max_val = agg['Tempo_Medio'].max()
        for bar, speedup in zip(bars, agg['Speedup_Medio']):
            width = bar.get_width()
            ax.text(width + (max_val * 0.02), bar.get_y() + bar.get_height()/2,
                    f"{width:.2f}s ({speedup:.2f}x)",
                    ha='left', va='center', fontsize=8.5, fontweight='bold', color='#1e293b')

        ax.set_title(res_t, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Tempo Médio (s)', fontsize=10)
        ax.set_xlim(0, max_val * 1.40)
        ax.grid(axis='x', alpha=0.6)

    # Legenda global
    legend_elements = [
        Patch(facecolor=CORES_SCHEDULE['dynamic'], label='Dynamic (Balanceamento Dinâmico)'),
        Patch(facecolor=CORES_SCHEDULE['guided'], label='Guided (Granularidade Decrescente)'),
        Patch(facecolor=CORES_SCHEDULE['static'], label='Static (Divisão Estática)'),
        Patch(facecolor=CORES_SCHEDULE['auto'], label='Auto (OpenMP Default)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), ncol=4, fontsize=9, framealpha=0.95)
    plt.tight_layout()

    salvar_figura(fig, f"tipo4_comparacao_escalonamento_resolucoes_{tag}", dir_png, dir_pdf, pdf_pages)


# ==============================================================================
# TIPO 5: ESCALABILIDADE DO TEMPO DE EXECUÇÃO EM FUNÇÃO DA RESOLUÇÃO (4K -> 8K -> 16K)
# ==============================================================================
def gerar_tipo5_escalabilidade_resolucao(df, dir_png, dir_pdf, pdf_pages=None):
    """
    Gera painel com a escalabilidade do tempo de execução em função do aumento
    da resolução da imagem (4096 -> 8192 -> 16384) para Visão Completa e Zoom.
    """
    scenarios = [
        {'name': 'Visão Completa', 'desc': 'Visão Completa (Max Iter: 1000)', 'tag': 'visao_completa'},
        {'name': 'Zoom (Seahorse)', 'desc': 'Zoom - Seahorse (Max Iter: 5000)', 'tag': 'zoom_seahorse'}
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle('Escalabilidade do Tempo de Execução em Função da Resolução da Imagem (4K → 8K → 16K)',
                 fontsize=13, fontweight='bold', y=1.02)

    x_labels = ['4K\n(4096×4096)', '8K\n(8192×8192)', '16K\n(16384×16384)']
    x_positions = [1, 2, 3]

    estrategias = [
        {'code': 'ponto_a_ponto', 'sched': None, 'chunk': None, 'threads': 1, 'label': '1 Thread (Ponto a Ponto - Base)', 'color': '#64748b', 'marker': 'D', 'ls': '--'},
        {'code': 'paralelo_collapse', 'sched': 'dynamic', 'chunk': 1024, 'threads': 4, 'label': 'Collapse - Dynamic (1024)', 'color': '#10b981', 'marker': 'o', 'ls': '-'},
        {'code': 'paralelo_collapse', 'sched': 'guided', 'chunk': 1024, 'threads': 4, 'label': 'Collapse - Guided (1024)', 'color': '#8b5cf6', 'marker': 'v', 'ls': '-'},
        {'code': 'paralelo', 'sched': 'dynamic', 'chunk': 64, 'threads': 4, 'label': '1D - Dynamic (64)', 'color': '#06b6d4', 'marker': '^', 'ls': '-'},
        {'code': 'paralelo', 'sched': 'static', 'chunk': 64, 'threads': 4, 'label': '1D - Static (64)', 'color': '#ef4444', 'marker': 's', 'ls': '-'},
    ]

    for ax, sc in zip(axes, scenarios):
        sub_sc = df[df['Scenario'] == sc['name']].copy()
        
        for est in estrategias:
            tempos = []
            valid = True
            for w in [4096, 8192, 16384]:
                if est['code'] == 'ponto_a_ponto':
                    cond = (sub_sc['WIDTH'] == w) & (sub_sc['Code'] == 'ponto_a_ponto')
                else:
                    cond = (sub_sc['WIDTH'] == w) & (sub_sc['Code'] == est['code']) & (sub_sc['Schedule'] == est['sched']) & (sub_sc['ChunkSize'] == est['chunk']) & (sub_sc['Threads'] == est['threads'])
                
                match = sub_sc[cond]
                if match.empty:
                    valid = False
                    break
                tempos.append(match['TempoGasto'].mean())

            if valid and len(tempos) == 3:
                ax.plot(x_positions, tempos, marker=est['marker'], color=est['color'],
                        linestyle=est['ls'], label=est['label'], linewidth=2.0, markersize=7)
                for x, t in zip(x_positions, tempos):
                    ax.annotate(f"{t:.1f}s", (x, t), textcoords="offset points",
                                xytext=(6, 4 if est['code']=='ponto_a_ponto' else -10),
                                fontsize=8, fontweight='bold', color=est['color'])

        ax.set_title(sc['desc'], fontsize=11.5, fontweight='bold')
        ax.set_xlabel('Resolução da Imagem', fontsize=10.5)
        ax.set_ylabel('Tempo de Execução Médio (segundos)', fontsize=10.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.grid(True)
        ax.legend(loc='upper left', fontsize=8.5)

    plt.tight_layout()
    salvar_figura(fig, "tipo5_escalabilidade_resolucao", dir_png, dir_pdf, pdf_pages)


# ==============================================================================
# EXECUÇÃO PRINCIPAL - EXPORTAÇÃO COMPLETA (PNG + PDF INDIVIDUAL + RELATÓRIO PDF)
# ==============================================================================
def gerar_todos_graficos(etapa=2, machine="deCasa", csv_path=None):
    print("=" * 80)
    etapa_str = f" - ETAPA {etapa}" if etapa is not None else ""
    print(f" 🎨 EXPORTANDO GRÁFICOS EM PDF E PNG (MANDELBROT OPENMP{etapa_str})")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    
    if etapa is not None:
        relatorios_dir = os.path.join(root_dir, "relatorios", f"etapa{etapa}")
    else:
        relatorios_dir = os.path.join(root_dir, "relatorios")

    dir_png = os.path.join(relatorios_dir, "png")
    dir_pdf = os.path.join(relatorios_dir, "pdf")
    os.makedirs(dir_png, exist_ok=True)
    os.makedirs(dir_pdf, exist_ok=True)

    relatorio_pdf_path = os.path.join(relatorios_dir, "relatorio_graficos_mandelbrot.pdf")
    df = carregar_dados(csv_path=csv_path, machine=machine, etapa=etapa)

    print(f"\n📁 Diretório PNG: {dir_png}")
    print(f"📁 Diretório PDF: {dir_pdf}")
    print(f"📑 Relatório PDF Consolidado: {relatorio_pdf_path}\n")

    with PdfPages(relatorio_pdf_path) as pdf_multipage:
        # 1. Gráficos individuais por cenário e resolução (Tipos 1, 2 e 3)
        for i, cenario in enumerate(CENARIOS_LISTA, 1):
            print(f"--- [Cenário {i}/6] {cenario['desc']} ---")
            # Tipo 1: Escalonamentos
            gerar_tipo1_schedules(df, cenario, dir_png, dir_pdf, pdf_multipage)
            # Tipo 2: Collapse vs 1D
            gerar_tipo2_collapse_vs_1d(df, cenario, dir_png, dir_pdf, pdf_multipage)
            # Tipo 3: Métricas (Tempo / Speedup / Eficiência 1 vs 4 Threads)
            gerar_tipo3_metricas(df, cenario, dir_png, dir_pdf, pdf_multipage)
            print()

        # 2. Gráficos consolidados longitudinais (Tipo 4: Comparação ao longo das resoluções)
        print("--- [Consolidado] Tipo 4: Escalonamentos ao Longo das Resoluções ---")
        gerar_tipo4_schedules_multi_res(
            df, 'Visão Completa', 'visao_completa',
            'Visão Completa (Max Iter: 1000)', dir_png, dir_pdf, pdf_multipage
        )
        gerar_tipo4_schedules_multi_res(
            df, 'Zoom (Seahorse)', 'zoom_seahorse',
            'Zoom - Seahorse (Max Iter: 5000)', dir_png, dir_pdf, pdf_multipage
        )
        print()

        # 3. Gráfico de escalabilidade em função da resolução (Tipo 5: 4K -> 8K -> 16K)
        print("--- [Consolidado] Tipo 5: Escalabilidade em Função da Resolução (4K → 8K → 16K) ---")
        gerar_tipo5_escalabilidade_resolucao(df, dir_png, dir_pdf, pdf_multipage)
        print()

    print("=" * 80)
    print("🎉 Exportação concluída com sucesso!")
    print(f"  ✅ PDFs individuais em '{dir_pdf}/'")
    print(f"  ✅ PNGs individuais em '{dir_png}/'")
    print(f"  ✅ Relatório PDF completo em '{relatorio_pdf_path}'")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geração e Exportação de Gráficos de Desempenho Paralelo")
    parser.add_argument("--etapa", type=int, default=2, help="Etapa do projeto a gerar gráficos (padrão: 2)")
    parser.add_argument("--csv", default=None, help="Caminho direto para o CSV de execuções (sobrescreve --etapa)")
    parser.add_argument("--machine", default="deCasa", help="Filtrar por nome de máquina (padrão: deCasa, use 'all' para todas)")
    args = parser.parse_args()
    gerar_todos_graficos(etapa=args.etapa, machine=args.machine, csv_path=args.csv)