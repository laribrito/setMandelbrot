#!/usr/bin/env python3
"""
Geração de Gráficos e Análise Comparativa MPI vs. OpenMP (Mandelbrot)
--------------------------------------------------------------------
Cruza os achados da Etapa 1 (Memória Compartilhada / OpenMP) com a Etapa 2
(Memória Distribuída / MPI), confrontando:
  1. Custo de particionamento isolado + troca explícita de mensagens (MPI)
     versus sincronização implícita de threads (OpenMP) via atômicos em RAM.
  2. Sensibilidade ao desbalanceamento de carga (Cenário Simétrico vs. Assimétrico).
  3. Aproveitamento do hardware (Intel Core i3-8130U: 2 núcleos físicos / 4 threads HT).
  4. Eficiência Paralela e Eficiência Física real.

Estilo Visual:
  - OpenMP destacado com hachuras densas ('///' e '\\\\') e bordas reforçadas.
  - MPI com preenchimento sólido institucional.
  - Tipografia ampliada, números maiores e badges com caixas de destaque.
  - Headroom ampliado nos eixos para evitar qualquer truncamento de rótulos.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Garante cache do matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

# Configuração visual e tipográfica moderna e limpa
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
plt.rcParams['axes.edgecolor'] = '#64748b'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['hatch.linewidth'] = 1.6

PALETA = {
    'OpenMP Collapse': '#059669',          # Verde esmeralda profundo
    'OpenMP 1D': '#0284c7',                # Azul ciano vibrante
    'MPI Estático Cíclico': '#2563eb',     # Azul real
    'MPI Dinâmico Meia-Linha': '#d97706',  # Âmbar / Laranja
    'MPI Dinâmico Dedicado': '#7c3aed',    # Violeta
    'MPI Estático Bloco': '#64748b',      # Cinza ardósia
    'MPI Dinâmico Participativo': '#dc2626'# Vermelho carmesim
}

def resolve_path(rel_path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    return os.path.normpath(os.path.join(root_dir, rel_path))

def carregar_dados():
    csv_path = resolve_path("data/dados_tratados_mpi.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo '{csv_path}' não encontrado.")
    
    df = pd.read_csv(csv_path)
    df_p4 = df[df["Threads"] == 4].copy()
    return df, df_p4

def consolidar_tabela_comparativa(df_p4: pd.DataFrame) -> pd.DataFrame:
    """Gera dataset consolidado com os melhores tempos de cada paradigma e estratégia."""
    registros = []
    cenarios = ["Visão Completa", "Zoom (Seahorse)"]
    resolucoes = [4096, 8192, 16384]
    
    for sc in cenarios:
        for w in resolucoes:
            sub = df_p4[(df_p4["Scenario"] == sc) & (df_p4["WIDTH"] == w)]
            t1 = sub["T1_Tratado"].iloc[0]
            
            omp_col = sub[sub["Code"] == "paralelo_collapse"].sort_values("Tempo_Tratado").iloc[0]
            omp_1d = sub[sub["Code"] == "paralelo"].sort_values("Tempo_Tratado").iloc[0]
            mpi_cyc = sub[(sub["Code"] == "mpi_estatico") & (sub["Schedule"] == "cyclic")].sort_values("Tempo_Tratado").iloc[0]
            mpi_blk = sub[(sub["Code"] == "mpi_estatico") & (sub["Schedule"] == "block")].sort_values("Tempo_Tratado").iloc[0]
            mpi_dyn = sub[sub["Code"] == "mpi_dinamico"].sort_values("Tempo_Tratado").iloc[0]
            mpi_hlf = sub[sub["Code"] == "mpi_dinamico_meia_linha"].sort_values("Tempo_Tratado").iloc[0]
            mpi_prt = sub[sub["Code"] == "mpi_dinamico_participativo"].sort_values("Tempo_Tratado").iloc[0]
            
            itens = [
                ("OpenMP Collapse", "Memória Compartilhada", omp_col),
                ("OpenMP 1D", "Memória Compartilhada", omp_1d),
                ("MPI Estático Cíclico", "Memória Distribuída", mpi_cyc),
                ("MPI Dinâmico Meia-Linha", "Memória Distribuída", mpi_hlf),
                ("MPI Dinâmico Dedicado", "Memória Distribuída", mpi_dyn),
                ("MPI Estático Bloco", "Memória Distribuída", mpi_blk),
                ("MPI Dinâmico Participativo", "Memória Distribuída", mpi_prt),
            ]
            
            for nome, paradigma, row in itens:
                tempo = row["Tempo_Tratado"]
                sp = row["Speedup_Tratado"]
                ep = row["Eficiencia_Tratada"]
                ep_fis = row["Eficiencia_Fisica"]
                ratio_vs_omp = tempo / omp_col["Tempo_Tratado"]
                
                registros.append({
                    "Cenário": sc,
                    "Resolução": f"{w}x{w}",
                    "WIDTH": w,
                    "Estratégia": nome,
                    "Paradigma": paradigma,
                    "Code": row["Code"],
                    "Schedule": row["Schedule"],
                    "ChunkSize": row["ChunkSize"],
                    "T1_s": t1,
                    "Tp_s": tempo,
                    "Speedup": sp,
                    "Eficiencia_p4_%": ep * 100.0,
                    "Eficiencia_Fisica_%": ep_fis * 100.0,
                    "Overhead_vs_OpenMP_Ratio": ratio_vs_omp
                })
                
    res_df = pd.DataFrame(registros)
    out_csv = resolve_path("data/analise_comparativa_mpi_openmp.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"✅ Salvo dataset consolidado: {out_csv}")
    return res_df

def plotar_speedup_comparativo(comp_df: pd.DataFrame):
    """
    Gráfico de barras comparando o Speedup de OpenMP vs. todas as variantes MPI.
    OpenMP estilizado com hachura densa ('///'), borda escura reforçada e números destacados em caixas legíveis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), sharey=True)
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]
    resolucoes = [4096, 8192, 16384]
    labels_res = ['4096 × 4096 (4K)', '8192 × 8192 (8K)', '16384 × 16384 (16K)']
    
    # Lista de estratégias com cor, hachura e borda diferenciada
    estrategias_config = [
        ('OpenMP Collapse', PALETA['OpenMP Collapse'], '///', '#064e3b', 2.0),
        ('MPI Estático Cíclico', PALETA['MPI Estático Cíclico'], None, '#1e3a8a', 0.8),
        ('MPI Dinâmico Dedicado', PALETA['MPI Dinâmico Dedicado'], None, '#4c1d95', 0.8),
        ('MPI Dinâmico Meia-Linha', PALETA['MPI Dinâmico Meia-Linha'], None, '#78350f', 0.8),
        ('MPI Estático Bloco', PALETA['MPI Estático Bloco'], None, '#334155', 0.8),
        ('MPI Dinâmico Participativo', PALETA['MPI Dinâmico Participativo'], None, '#7f1d1d', 0.8),
    ]
    
    width = 0.135
    
    for cenario_nome, ax in cenarios:
        sub_c = comp_df[comp_df["Cenário"] == cenario_nome]
        x = np.arange(len(resolucoes))
        
        for i, (est, cor, hatch, edge_col, lw) in enumerate(estrategias_config):
            sub_est = sub_c[sub_c["Estratégia"] == est].sort_values("WIDTH")
            sp_vals = sub_est["Speedup"].values
            
            offset = (i - 2.5) * width
            label_tag = f"★ {est} (Mem. Compartilhada)" if "OpenMP" in est else f"{est} (MPI)"
            
            rects = ax.bar(
                x + offset, sp_vals, width,
                label=label_tag,
                color=cor,
                hatch=hatch,
                edgecolor=edge_col,
                linewidth=lw,
                alpha=0.95
            )
            
            # Anotações numéricas grandes, legíveis e com badge protetor
            is_openmp = "OpenMP" in est
            for rect in rects:
                h = rect.get_height()
                
                # Cores e destaque para os números
                if is_openmp:
                    box_style = dict(boxstyle='round,pad=0.28', facecolor='#d1fae5', edgecolor='#059669', linewidth=1.5)
                    txt_color = '#064e3b'
                    fsize = 10.5
                    fweight = 'black'
                else:
                    box_style = dict(boxstyle='round,pad=0.20', facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=0.8, alpha=0.92)
                    txt_color = '#0f172a'
                    fsize = 9.0
                    fweight = 'bold'
                
                ax.annotate(
                    f'{h:.2f}×',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=fsize,
                    fontweight=fweight,
                    color=txt_color,
                    bbox=box_style,
                    rotation=30 if not is_openmp else 0
                )
        
        # Linhas de referência teóricas com tipografia aprimorada
        ax.axhline(1.0, color='#64748b', linestyle=':', linewidth=1.8, label='Baseline Serial (1.00×)')
        ax.axhline(2.0, color='#0284c7', linestyle='-.', linewidth=1.8, label='Teto 2 Núcleos Físicos (2.00×)')
        ax.axhline(4.0, color='#dc2626', linestyle='--', linewidth=2.0, label='Speedup Ideal Linear (4.00×)')
        
        ax.set_title(f'Cenário: {cenario_nome}', fontsize=15, fontweight='bold', pad=14, color='#0f172a')
        ax.set_xlabel('Resolução da Matriz', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_res, fontsize=11.5, fontweight='bold')
        ax.set_ylim(0, 5.0)  # Headroom confortável para não cortar números
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    axes[0].set_ylabel('Speedup Real ($S_p = T_1 / T_p$)', fontsize=13, fontweight='bold', labelpad=10)
    
    # Legenda ampla e visualmente organizada
    axes[1].legend(
        loc='upper right',
        frameon=True,
        framealpha=0.96,
        facecolor='#ffffff',
        edgecolor='#94a3b8',
        fontsize=9.5,
        ncol=2
    )
    
    plt.suptitle(
        'Confronto de Escalabilidade: OpenMP (Memória Compartilhada) vs. MPI (Memória Distribuída)\n'
        'Hardware: Intel Core i3-8130U (2 Cores Físicos / 4 Threads HT) — p = 4 unidades de processamento',
        fontsize=16, fontweight='bold', y=0.99, color='#0f172a'
    )
    plt.tight_layout()
    
    out_png = resolve_path("relatorios/etapa2/png/mpi_vs_openmp_speedup_comparativo.png")
    out_pdf = resolve_path("relatorios/etapa2/pdf/mpi_vs_openmp_speedup_comparativo.pdf")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"✅ Salvo gráfico com alta legibilidade: {out_png}")

def plotar_confronto_dinamico(df_p4: pd.DataFrame):
    """
    Confronto direto dos modelos DINÂMICOS em escala logarítmica.
    Destaca OpenMP Collapse e OpenMP 1D com hachuras densas ('///' e '\\\\').
    Números ampliados e legíveis com badges arredondados.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), sharey=False)
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]
    resolucoes = [4096, 8192, 16384]
    labels_res = ['4096 (4K)', '8192 (8K)', '16384 (16K)']
    
    for cenario_nome, ax in cenarios:
        sub_c = df_p4[df_p4["Scenario"] == cenario_nome]
        
        omp_col_dyn = []
        omp_1d_dyn = []
        mpi_dyn = []
        mpi_hlf = []
        mpi_prt = []
        
        for w in resolucoes:
            s_w = sub_c[sub_c["WIDTH"] == w]
            
            c_dyn = s_w[(s_w["Code"] == "paralelo_collapse") & (s_w["Schedule"] == "dynamic")].sort_values("Tempo_Tratado")
            omp_col_dyn.append(c_dyn["Tempo_Tratado"].iloc[0] if not c_dyn.empty else np.nan)
            
            o_dyn = s_w[(s_w["Code"] == "paralelo") & (s_w["Schedule"] == "dynamic")].sort_values("Tempo_Tratado")
            omp_1d_dyn.append(o_dyn["Tempo_Tratado"].iloc[0] if not o_dyn.empty else np.nan)
            
            m_dyn = s_w[s_w["Code"] == "mpi_dinamico"].sort_values("Tempo_Tratado")
            mpi_dyn.append(m_dyn["Tempo_Tratado"].iloc[0] if not m_dyn.empty else np.nan)
            
            m_hlf = s_w[s_w["Code"] == "mpi_dinamico_meia_linha"].sort_values("Tempo_Tratado")
            mpi_hlf.append(m_hlf["Tempo_Tratado"].iloc[0] if not m_hlf.empty else np.nan)
            
            m_prt = s_w[s_w["Code"] == "mpi_dinamico_participativo"].sort_values("Tempo_Tratado")
            mpi_prt.append(m_prt["Tempo_Tratado"].iloc[0] if not m_prt.empty else np.nan)
            
        x = np.arange(len(resolucoes))
        width = 0.155
        
        # Barras OpenMP hachuradas e com bordas fortes
        b1 = ax.bar(x - 2.0*width, omp_col_dyn, width,
                    label='★ OpenMP Collapse (Dynamic 2D)',
                    color=PALETA['OpenMP Collapse'], hatch='///', edgecolor='#064e3b', linewidth=2.0, alpha=0.95)
        
        b2 = ax.bar(x - 1.0*width, omp_1d_dyn, width,
                    label='★ OpenMP 1D (Dynamic)',
                    color=PALETA['OpenMP 1D'], hatch='\\\\', edgecolor='#0c4a6e', linewidth=1.8, alpha=0.95)
        
        # Barras MPI sólidas
        b3 = ax.bar(x, mpi_hlf, width,
                    label='MPI Dinâmico Meia-Linha',
                    color=PALETA['MPI Dinâmico Meia-Linha'], edgecolor='#78350f', linewidth=0.8, alpha=0.92)
        
        b4 = ax.bar(x + 1.0*width, mpi_dyn, width,
                    label='MPI Dinâmico Dedicado (Worker-Only)',
                    color=PALETA['MPI Dinâmico Dedicado'], edgecolor='#4c1d95', linewidth=0.8, alpha=0.92)
        
        b5 = ax.bar(x + 2.0*width, mpi_prt, width,
                    label='MPI Dinâmico Participativo (Starvation)',
                    color=PALETA['MPI Dinâmico Participativo'], edgecolor='#7f1d1d', linewidth=0.8, alpha=0.92)
        
        grupos = [
            (b1, True, '#d1fae5', '#059669', '#064e3b'),
            (b2, True, '#e0f2fe', '#0284c7', '#0c4a6e'),
            (b3, False, '#ffffff', '#cbd5e1', '#0f172a'),
            (b4, False, '#ffffff', '#cbd5e1', '#0f172a'),
            (b5, False, '#fee2e2', '#ef4444', '#7f1d1d')
        ]
        
        for bars, is_omp, f_col, b_col, t_col in grupos:
            for rect in bars:
                h = rect.get_height()
                if not np.isnan(h) and h > 0:
                    box_prop = dict(boxstyle='round,pad=0.25', facecolor=f_col, edgecolor=b_col, linewidth=1.2 if is_omp else 0.8)
                    rot = 0 if (is_omp or h >= 100) else 25
                    f_size = 9.8 if is_omp else 8.5
                    f_weight = 'black' if is_omp else 'bold'
                    
                    ax.annotate(
                        f'{h:.2f}s' if h < 10 else f'{h:.1f}s',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=f_size,
                        fontweight=f_weight,
                        color=t_col,
                        bbox=box_prop,
                        rotation=rot
                    )
                    
        ax.set_title(f'Cenário: {cenario_nome}', fontsize=15, fontweight='bold', pad=14, color='#0f172a')
        ax.set_xlabel('Resolução', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_res, fontsize=11.5, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(0.5, 950)  # Headroom amplo para a escala log
        ax.set_ylabel('Tempo de Execução $T_p$ (s) — Escala Logarítmica', fontsize=13, fontweight='bold', labelpad=10)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, which="both", linestyle='--', alpha=0.55)
        
    axes[1].legend(
        loc='upper left',
        frameon=True,
        framealpha=0.96,
        facecolor='#ffffff',
        edgecolor='#94a3b8',
        fontsize=9.5
    )
    
    plt.suptitle(
        'Confronto dos Mecanismos Dinâmicos: Sincronização Implícita (OpenMP) vs. Troca de Mensagens (MPI)\n'
        'Evidenciando ordens de grandeza de penalidade por IPC e colapso por inanição (escala logarítmica)',
        fontsize=16, fontweight='bold', y=0.99, color='#0f172a'
    )
    plt.tight_layout()
    
    out_png = resolve_path("relatorios/etapa2/png/confronto_dinamico_openmp_vs_mpi.png")
    out_pdf = resolve_path("relatorios/etapa2/pdf/confronto_dinamico_openmp_vs_mpi.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"✅ Salvo gráfico com alta legibilidade: {out_png}")

def plotar_eficiencia_fisica_comparativa(comp_df: pd.DataFrame):
    """
    Gráfico de Eficiência Real sobre os 2 núcleos físicos da máquina.
    OpenMP destacado com hachura '///' e números grandes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.0), sharey=True)
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]
    resolucoes = [4096, 8192, 16384]
    labels_res = ['4096 (4K)', '8192 (8K)', '16384 (16K)']
    
    estrategias_config = [
        ('OpenMP Collapse', PALETA['OpenMP Collapse'], '///', '#064e3b', 2.0),
        ('MPI Estático Cíclico', PALETA['MPI Estático Cíclico'], None, '#1e3a8a', 0.8),
        ('MPI Dinâmico Meia-Linha', PALETA['MPI Dinâmico Meia-Linha'], None, '#78350f', 0.8),
        ('MPI Dinâmico Dedicado', PALETA['MPI Dinâmico Dedicado'], None, '#4c1d95', 0.8),
        ('MPI Estático Bloco', PALETA['MPI Estático Bloco'], None, '#334155', 0.8),
    ]
    width = 0.155
    
    for cenario_nome, ax in cenarios:
        sub_c = comp_df[comp_df["Cenário"] == cenario_nome]
        x = np.arange(len(resolucoes))
        
        for i, (est, cor, hatch, edge_col, lw) in enumerate(estrategias_config):
            sub_est = sub_c[sub_c["Estratégia"] == est].sort_values("WIDTH")
            ef_vals = sub_est["Eficiencia_Fisica_%"].values
            
            offset = (i - 2) * width
            label_tag = f"★ {est} (Mem. Compartilhada)" if "OpenMP" in est else f"{est} (MPI)"
            
            rects = ax.bar(
                x + offset, ef_vals, width,
                label=label_tag,
                color=cor,
                hatch=hatch,
                edgecolor=edge_col,
                linewidth=lw,
                alpha=0.95
            )
            
            is_openmp = "OpenMP" in est
            for rect in rects:
                h = rect.get_height()
                if is_openmp:
                    box_prop = dict(boxstyle='round,pad=0.28', facecolor='#d1fae5', edgecolor='#059669', linewidth=1.5)
                    txt_col = '#064e3b'
                    f_size = 10.5
                    f_weight = 'black'
                else:
                    box_prop = dict(boxstyle='round,pad=0.20', facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=0.8, alpha=0.92)
                    txt_col = '#0f172a'
                    f_size = 9.0
                    f_weight = 'bold'
                    
                ax.annotate(
                    f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=f_size,
                    fontweight=f_weight,
                    color=txt_col,
                    bbox=box_prop,
                    rotation=25 if not is_openmp else 0
                )
                
        ax.axhline(100.0, color='#dc2626', linestyle='--', linewidth=2.0, label='Limite Teórico 2 Cores Físicos (100%)')
        ax.axhline(50.0, color='#64748b', linestyle=':', linewidth=1.5, label='Capacidade de 1 Core Físico (50%)')
        
        ax.set_title(f'Cenário: {cenario_nome}', fontsize=15, fontweight='bold', pad=14, color='#0f172a')
        ax.set_xlabel('Resolução', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_res, fontsize=11.5, fontweight='bold')
        ax.set_ylim(0, 235)  # Headroom confortável para até 186%
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    axes[0].set_ylabel('Eficiência Real sobre Núcleos Físicos ($S_p / 2$ %)', fontsize=13, fontweight='bold', labelpad=10)
    axes[1].legend(
        loc='upper right',
        frameon=True,
        framealpha=0.96,
        facecolor='#ffffff',
        edgecolor='#94a3b8',
        fontsize=9.5,
        ncol=2
    )
    
    plt.suptitle(
        'Eficiência Paralela sobre Núcleos Físicos: Aproveitamento de SMT (Hyper-Threading)\n'
        'OpenMP atinge até 186% de eficiência física contra 109% do melhor MPI graças ao compartilhamento de L3',
        fontsize=16, fontweight='bold', y=0.99, color='#0f172a'
    )
    plt.tight_layout()
    
    out_png = resolve_path("relatorios/etapa2/png/mpi_vs_openmp_eficiencia_fisica.png")
    out_pdf = resolve_path("relatorios/etapa2/pdf/mpi_vs_openmp_eficiencia_fisica.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"✅ Salvo gráfico com alta legibilidade: {out_png}")

def plotar_tempos_execucao_comparativo(comp_df: pd.DataFrame):
    """
    Gráfico adicional comparando o Tempo de Execução Tp (s) bruto com OpenMP hachurado.
    Permite visualizar a queda dramática de tempo oferecida pelo OpenMP.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.0), sharey=False)
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]
    resolucoes = [4096, 8192, 16384]
    labels_res = ['4096 (4K)', '8192 (8K)', '16384 (16K)']
    
    estrategias_config = [
        ('OpenMP Collapse', PALETA['OpenMP Collapse'], '///', '#064e3b', 2.0),
        ('MPI Estático Cíclico', PALETA['MPI Estático Cíclico'], None, '#1e3a8a', 0.8),
        ('MPI Dinâmico Meia-Linha', PALETA['MPI Dinâmico Meia-Linha'], None, '#78350f', 0.8),
        ('MPI Dinâmico Dedicado', PALETA['MPI Dinâmico Dedicado'], None, '#4c1d95', 0.8),
        ('MPI Estático Bloco', PALETA['MPI Estático Bloco'], None, '#334155', 0.8),
    ]
    width = 0.155
    
    for cenario_nome, ax in cenarios:
        sub_c = comp_df[comp_df["Cenário"] == cenario_nome]
        x = np.arange(len(resolucoes))
        
        for i, (est, cor, hatch, edge_col, lw) in enumerate(estrategias_config):
            sub_est = sub_c[sub_c["Estratégia"] == est].sort_values("WIDTH")
            t_vals = sub_est["Tp_s"].values
            
            offset = (i - 2) * width
            label_tag = f"★ {est} (Mem. Compartilhada)" if "OpenMP" in est else f"{est} (MPI)"
            
            rects = ax.bar(
                x + offset, t_vals, width,
                label=label_tag,
                color=cor,
                hatch=hatch,
                edgecolor=edge_col,
                linewidth=lw,
                alpha=0.95
            )
            
            is_openmp = "OpenMP" in est
            for rect in rects:
                h = rect.get_height()
                if is_openmp:
                    box_prop = dict(boxstyle='round,pad=0.28', facecolor='#d1fae5', edgecolor='#059669', linewidth=1.5)
                    txt_col = '#064e3b'
                    f_size = 10.2
                    f_weight = 'black'
                else:
                    box_prop = dict(boxstyle='round,pad=0.20', facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=0.8, alpha=0.92)
                    txt_col = '#0f172a'
                    f_size = 8.8
                    f_weight = 'bold'
                    
                ax.annotate(
                    f'{h:.2f}s' if h < 10 else f'{h:.1f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=f_size,
                    fontweight=f_weight,
                    color=txt_col,
                    bbox=box_prop,
                    rotation=25 if not is_openmp else 0
                )
                
        ax.set_title(f'Cenário: {cenario_nome}', fontsize=15, fontweight='bold', pad=14, color='#0f172a')
        ax.set_xlabel('Resolução', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_res, fontsize=11.5, fontweight='bold')
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Ajusta headroom
        max_val = sub_c[sub_c["Estratégia"].isin([e[0] for e in estrategias_config])]["Tp_s"].max()
        ax.set_ylim(0, max_val * 1.25)
        
    axes[0].set_ylabel('Tempo de Execução $T_p$ (segundos)', fontsize=13, fontweight='bold', labelpad=10)
    axes[1].legend(
        loc='upper left',
        frameon=True,
        framealpha=0.96,
        facecolor='#ffffff',
        edgecolor='#94a3b8',
        fontsize=9.5,
        ncol=2
    )
    
    plt.suptitle(
        'Tempo de Execução Paralelo $T_p$ (s): OpenMP vs. Estratégias MPI\n'
        'Menor é melhor — OpenMP reduz o tempo em até 2,5× em relação ao MPI dinâmico e 3,8× vs. bloco',
        fontsize=16, fontweight='bold', y=0.99, color='#0f172a'
    )
    plt.tight_layout()
    
    out_png = resolve_path("relatorios/etapa2/png/mpi_vs_openmp_tempos_execucao.png")
    out_pdf = resolve_path("relatorios/etapa2/pdf/mpi_vs_openmp_tempos_execucao.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"✅ Salvo gráfico de tempos de execução: {out_png}")

def main():
    df, df_p4 = carregar_dados()
    comp_df = consolidar_tabela_comparativa(df_p4)
    plotar_speedup_comparativo(comp_df)
    plotar_confronto_dinamico(df_p4)
    plotar_eficiencia_fisica_comparativa(comp_df)
    plotar_tempos_execucao_comparativo(comp_df)
    print("🚀 Gráficos de alta legibilidade gerados com sucesso em PNG e PDF!")

if __name__ == "__main__":
    main()
