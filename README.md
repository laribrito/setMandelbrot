# Benchmark de Geração do Conjunto de Mandelbrot com OpenMP

Projeto de avaliação de desempenho, escalabilidade e políticas de escalonamento paralelo utilizando **OpenMP** e **C++17**, com suíte de automação, métricas e geração de gráficos em **Python**.

> **Instituição**: Universidade Estadual de Santa Cruz (UESC)  
> **Disciplina**: DEC107 — Processamento Paralelo  
> **Autora**: Larissa de Brito Santos  

---

## 📁 Estrutura do Repositório

```text
setMandel/
├── src/                          # Códigos-fonte em C++17
│   ├── sequencial.cpp            # Versão sequencial com otimização geométrica (simetria)
│   ├── ponto_a_ponto.cpp         # Versão sequencial pura (baseline de referência)
│   ├── paralelo.cpp              # Paralelismo 1D no loop externo (#pragma omp parallel for)
│   ├── paralelo_collapse.cpp     # Paralelismo 2D (#pragma omp parallel for collapse(2))
│   └── comparador.cpp            # Utilitário de validação e equivalência de matrizes PPM
│
├── scripts/                      # Suíte de automação, métricas e gráficos em Python
│   ├── compile.py                # Script de compilação multiplataforma (Linux / Windows)
│   ├── run_tests.py              # Automação de baterias de testes e histórico incremental
│   ├── metricas.py               # Processamento estatístico, cálculo de Speedup e Eficiência
│   └── graficos.py               # Renderização de gráficos individuais (PNG/PDF) e compilado
│
├── data/                         # Bases de dados e análises em CSV
│   ├── dateTimeExecution.csv     # Registro histórico completo de todas as execuções
│   ├── dados_com_metricas.csv    # Dataset consolidado com Speedup ($S_p$) e Eficiência ($E_p$)
│   ├── analise_schedules.csv     # Comparativo de escalonamentos (Static, Dynamic, Guided, Auto)
│   ├── analise_collapse_vs_1d.csv# Comparativo direto entre loop 1D e collapse(2)
│   ├── analise_escalabilidade.csv# Escalabilidade forte por número de threads (1, 2, 4)
│   └── analise_melhores_configs.csv # Ranking das configurações ótimas por resolução e cenário
│
├── relatorios/                   # Relatórios acadêmicos e gráficos gerados
│   ├── relatorio1.pdf            # Relatório técnico oficial da Etapa 1
│   ├── relatorio_graficos_mandelbrot.pdf # Compilado multi-páginas de gráficos
│   ├── pdf/                      # 18 gráficos vetoriais individuais (.pdf)
│   ├── png/                      # 18 gráficos em alta definição a 300 DPI (.png)
│   └── legados/                  # Gráficos anteriores e formatos auxiliares
│
├── requirements.txt              # Dependências Python (pandas, numpy, matplotlib)
├── in.txt                        # Arquivo de configuração de entrada padrão
└── .gitignore                    # Regras de exclusão de binários e ambientes virtuais
```

---

## 🚀 Como Compilar e Executar

### 1. Pré-requisitos e Dependências Python
Para utilizar os scripts de métricas e geração de gráficos, instale as dependências com:

```bash
pip install -r requirements.txt
```

### 2. Compilação dos Programas C++
Compile todos os programas C++ através do script Python multiplataforma:

```bash
python3 scripts/compile.py
```

### 3. Executando um Teste Manual
Configure os parâmetros no arquivo `in.txt` e execute o binário desejado:

```bash
./paralelo
# ou
./paralelo_collapse
```

Os resultados de tempo e parâmetros serão automaticamente registrados em `data/dateTimeExecution.csv` e a imagem gerada será salva em `out/`.

### 3. Automação de Baterias de Benchmarks
Para rodar todos os cenários (4096, 8192, 16384 em Full e Zoom) com 3 repetições:

```bash
python3 scripts/run_tests.py --tudo --threads 4
# ou pelo menu interativo:
python3 scripts/run_tests.py
```

### 4. Análise de Métricas (Speedup e Eficiência)
Para processar os dados do CSV, calcular Speedup/Eficiência e exportar tabelas comparativas:

```bash
python3 scripts/metricas.py --export-csv
```

### 5. Geração de Gráficos e Relatório PDF
Para gerar todos os 18 gráficos individuais (PNG/PDF) e o relatório compilado multi-páginas:

```bash
python3 scripts/graficos.py
```

Os arquivos serão exportados para a pasta `relatorios/`.
