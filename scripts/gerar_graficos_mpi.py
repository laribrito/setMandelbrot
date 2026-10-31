#!/usr/bin/env python3
"""
Gera gráficos de alta resolução para o Relatório 2 (MPI - Mandelbrot)
- Desempenho, Speedup e Eficiência para as 5 estratégias
- Análise de sensibilidade ao CHUNK_SIZE (1, 2, 4) sob diferentes resoluções
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuração estética moderna
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#94a3b8'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

CORES_ESTRATEGIAS = {
    'Bloco Contíguo': '#64748b',       # Cinza ardósia
    'Cíclica': '#2563eb',              # Azul real
    'Coordenador Dedicado': '#059669',  # Verde esmeralda
    'Participativo': '#dc2626',        # Vermelho carmesim
    'Meia-Linha': '#d97706'            # Âmbar / Laranja
}

def carregar_dados():
    df = pd.read_csv("data/dados_tratados_mpi.csv")
    mpi = df[df["Code"].str.startswith("mpi_")].copy()
    
    def map_strat(row):
        code = row["Code"]
        sched = row["Schedule"]
        if code == "mpi_estatico" and sched == "block":
            return "Bloco Contíguo"
        elif code == "mpi_estatico" and sched == "cyclic":
            return "Cíclica"
        elif code == "mpi_dinamico":
            return "Coordenador Dedicado"
        elif code == "mpi_dinamico_participativo":
            return "Participativo"
        elif code == "mpi_dinamico_meia_linha":
            return "Meia-Linha"
        return code

    mpi["Estrategia"] = mpi.apply(map_strat, axis=1)
    return mpi

def plotar_speedup_geral(df):
    """Gráfico de barras comparativo de Speedup para as 5 estratégias (usando o melhor chunk de cada)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]
    resolucoes = [4096, 8192, 16384]
    labels_res = ['4096 (4K)', '8192 (8K)', '16384 (16K)']
    
    estrategias = ['Cíclica', 'Meia-Linha', 'Coordenador Dedicado', 'Bloco Contíguo', 'Participativo']
    
    for cenario_nome, ax in cenarios:
        sub_c = df[df["Scenario"] == cenario_nome]
        x = np.arange(len(resolucoes))
        width = 0.16
        
        for i, est in enumerate(estrategias):
            sp_vals = []
            for w in resolucoes:
                sub_w = sub_c[(sub_c["WIDTH"] == w) & (sub_c["Estrategia"] == est)]
                # Pega a média/melhor speedup
                sp_vals.append(sub_w["Speedup_Tratado"].max())
            
            offset = (i - 2) * width
            rects = ax.bar(x + offset, sp_vals, width, label=est, color=CORES_ESTRATEGIAS[est], alpha=0.9, edgecolor='black', linewidth=0.5)
            
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.2f}x',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.axhline(1.0, color='#94a3b8', linestyle=':', linewidth=1.5, label='Speedup Base (1x - Serial)')
        ax.axhline(4.0, color='#ef4444', linestyle='--', linewidth=1.5, label='Speedup Ideal (4x - Linear)')
        
        ax.set_title(f'Cenário: {cenario_nome}', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Resolução da Matriz', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_res, fontsize=10)
        ax.set_ylim(0, 4.3)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    axes[0].set_ylabel('Speedup ($S_p = T_1 / T_p$)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=9)
    
    plt.suptitle('Comparativo de Speedup Máximo por Estratégia MPI (p = 4 processos)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs("relatorios/png", exist_ok=True)
    os.makedirs("relatorios/pdf", exist_ok=True)
    fig.savefig("relatorios/png/mpi_speedup_comparativo.png", dpi=300)
    fig.savefig("relatorios/pdf/mpi_speedup_comparativo.pdf")
    plt.close(fig)
    print(" Salvo: relatorios/png/mpi_speedup_comparativo.png")

def plotar_impacto_chunk_size(df):
    """Gráfico detalhado mostrando a sensibilidade do CHUNK_SIZE para cada estratégia e resolução."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    
    cenarios = ["Visão Completa", "Zoom (Seahorse)"]
    resolucoes = [4096, 8192, 16384]
    
    # Estratégias que variam chunk size (exclui Bloco que é chunk 0)
    estrategias_chunk = ['Cíclica', 'Coordenador Dedicado', 'Meia-Linha', 'Participativo']
    marcadores = {'Cíclica': 'o', 'Coordenador Dedicado': 's', 'Meia-Linha': '^', 'Participativo': 'D'}
    
    for row_idx, sc in enumerate(cenarios):
        for col_idx, w in enumerate(resolucoes):
            ax = axes[row_idx, col_idx]
            sub = df[(df["Scenario"] == sc) & (df["WIDTH"] == w)]
            
            for est in estrategias_chunk:
                sub_est = sub[sub["Estrategia"] == est].sort_values("ChunkSize")
                chunks = sub_est["ChunkSize"].values
                tempos = sub_est["Tempo_Tratado"].values
                
                ax.plot(chunks, tempos, marker=marcadores[est], linewidth=2.2, markersize=7,
                        color=CORES_ESTRATEGIAS[est], label=est)
                
                for x, y in zip(chunks, tempos):
                    ax.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", xytext=(0, 7),
                                ha='center', fontsize=8, fontweight='bold', color=CORES_ESTRATEGIAS[est])
            
            # Linha horizontal do Bloco Estático para referência
            sub_block = sub[sub["Estrategia"] == "Bloco Contíguo"]
            if not sub_block.empty:
                t_block = sub_block["Tempo_Tratado"].iloc[0]
                ax.axhline(t_block, color=CORES_ESTRATEGIAS['Bloco Contíguo'], linestyle='--', linewidth=1.5,
                           label=f'Bloco ({t_block:.2f}s)')
            
            ax.set_title(f'{sc} - {w}×{w}', fontsize=12, fontweight='bold')
            ax.set_xticks([1, 2, 4])
            ax.set_xticklabels(['Chunk 1', 'Chunk 2', 'Chunk 4'], fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if col_idx == 0:
                ax.set_ylabel('Tempo de Execução $T_p$ (s)', fontsize=11, fontweight='bold')
            if row_idx == 1:
                ax.set_xlabel('Tamanho do Bloco (Chunk Size)', fontsize=11, fontweight='bold')
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', fontsize=8, frameon=True)

    plt.suptitle('Sensibilidade ao CHUNK_SIZE: Tempo de Execução $T_p$ (s) por Resolução e Cenário', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig("relatorios/png/mpi_sensibilidade_chunk_size.png", dpi=300)
    fig.savefig("relatorios/pdf/mpi_sensibilidade_chunk_size.pdf")
    plt.close(fig)
    print(" Salvo: relatorios/png/mpi_sensibilidade_chunk_size.png")

def plotar_eficiencia_grid(df):
    """Gráfico de barras agrupadas mostrando a Eficiência (%) para todas as estratégias e resoluções."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]
    resolucoes = [4096, 8192, 16384]
    labels_res = ['4K', '8K', '16K']
    estrategias = ['Cíclica', 'Meia-Linha', 'Coordenador Dedicado', 'Bloco Contíguo', 'Participativo']
    
    for cenario_nome, ax in cenarios:
        sub_c = df[df["Scenario"] == cenario_nome]
        x = np.arange(len(resolucoes))
        width = 0.16
        
        for i, est in enumerate(estrategias):
            ef_vals = []
            for w in resolucoes:
                sub_w = sub_c[(sub_c["WIDTH"] == w) & (sub_c["Estrategia"] == est)]
                ef_vals.append(sub_w["Eficiencia_Tratada"].max() * 100)
            
            offset = (i - 2) * width
            rects = ax.bar(x + offset, ef_vals, width, label=est, color=CORES_ESTRATEGIAS[est], alpha=0.9, edgecolor='black', linewidth=0.5)
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
                
        ax.axhline(100.0, color='#ef4444', linestyle='--', linewidth=1.5, label='Eficiência Ideal (100%)')
        ax.axhline(50.0, color='#94a3b8', linestyle=':', linewidth=1.2, label='Patamar 50%')
        ax.set_title(f'Cenário: {cenario_nome}', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Resolução', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_res, fontsize=10)
        ax.set_ylim(0, 110)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    axes[0].set_ylabel('Eficiência Paralela ($E_p = S_p / p$ %)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=9)
    plt.suptitle('Comparativo de Eficiência Paralela por Estratégia MPI (p = 4)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig("relatorios/png/mpi_eficiencia_comparativo.png", dpi=300)
    fig.savefig("relatorios/pdf/mpi_eficiencia_comparativo.pdf")
    plt.close(fig)
    print(" Salvo: relatorios/png/mpi_eficiencia_comparativo.png")

def plotar_eficiencia_fisica_2nucleos(df):
    """Gráfico de Eficiência considerando os 2 núcleos físicos reais (Intel Core i3-8130U com 2 núcleos / 4 threads HT)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cenarios = [("Visão Completa", axes[0]), ("Zoom (Seahorse)", axes[1])]
    resolucoes = [4096, 8192, 16384]
    labels_res = ['4K', '8K', '16K']
    estrategias = ['Cíclica', 'Meia-Linha', 'Coordenador Dedicado', 'Bloco Contíguo', 'Participativo']
    
    for cenario_nome, ax in cenarios:
        sub_c = df[df["Scenario"] == cenario_nome]
        x = np.arange(len(resolucoes))
        width = 0.16
        
        for i, est in enumerate(estrategias):
            ef_vals = []
            for w in resolucoes:
                sub_w = sub_c[(sub_c["WIDTH"] == w) & (sub_c["Estrategia"] == est)]
                # Eficiência com base em 2 núcleos físicos: Sp / 2
                sp_max = sub_w["Speedup_Tratado"].max()
                ef_vals.append((sp_max / 2.0) * 100)
            
            offset = (i - 2) * width
            rects = ax.bar(x + offset, ef_vals, width, label=est, color=CORES_ESTRATEGIAS[est], alpha=0.9, edgecolor='black', linewidth=0.5)
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
                
        ax.axhline(100.0, color='#ef4444', linestyle='--', linewidth=1.5, label='Limite Ideal 2 Núcleos (100%)')
        ax.axhline(50.0, color='#94a3b8', linestyle=':', linewidth=1.2, label='1 Núcleo Físico (50%)')
        ax.set_title(f'Cenário: {cenario_nome}', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Resolução', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_res, fontsize=10)
        ax.set_ylim(0, 130)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    axes[0].set_ylabel('Eficiência Real sobre Núcleos Físicos ($S_p / 2$ %)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=9)
    plt.suptitle('Eficiência Paralela sobre Núcleos Físicos (Base p = 2 Cores Físicos / HT)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig("relatorios/png/mpi_eficiencia_fisica_2nucleos.png", dpi=300)
    fig.savefig("relatorios/pdf/mpi_eficiencia_fisica_2nucleos.pdf")
    plt.close(fig)
    print(" Salvo: relatorios/png/mpi_eficiencia_fisica_2nucleos.png")

if __name__ == "__main__":
    df = carregar_dados()
    plotar_speedup_geral(df)
    plotar_impacto_chunk_size(df)
    plotar_eficiencia_grid(df)
    plotar_eficiencia_fisica_2nucleos(df)
    print(" Gráficos de alta qualidade gerados em relatorios/png e relatorios/pdf!")
