# 📋 Mapeamento Completo dos 78 Casos de Teste de Corretude MPI (Etapa 2)

Este documento descreve detalhadamente a composição e a justificativa de cada um dos **78 casos de teste** executados pela suíte de validação de corretude ([scripts/test_corretude_mpi.py](file:///home/debrito/Programação/setMandel/scripts/test_corretude_mpi.py)) para o **Relatório 2 (MPI)** da disciplina **DEC107 — Processamento Paralelo (UESC)**.

---

## 🧮 1. Como chegamos ao número 78? (Equação da Matriz de Testes)

A matriz de testes foi projetada para garantir cobertura exaustiva de todas as combinações de:
- **2 Cenários do Plano Complexo** (Visão Completa e Zoom Profundo).
- **5 Estratégias Algorítmicas MPI**.
- **3 Variações de Granularidade (*chunk size*)** para cada estratégia (exceto Bloco, que divide em faixas contíguas).
- **3 Topologias de Processos** ($np = 2, 3, 4$), incluindo processos pares e ímpares.

$$\text{Total por Cenário} = \underbrace{3}_{\text{Estático Bloco}} + \underbrace{9}_{\text{Estático Cíclico}} + \underbrace{9}_{\text{Dinâmico Dedicado}} + \underbrace{9}_{\text{Dinâmico Participativo}} + \underbrace{9}_{\text{Dinâmico Meia-Linha}} = 39 \text{ testes}$$

$$\text{Total Global} = 39 \text{ testes} \times 2 \text{ cenários} = \mathbf{78 \text{ casos de teste}}$$

---

## 🌐 2. Definição dos Cenários Avaliados

Cada caso foi testado na resolução de **$512 \times 512$** ($262.144$ pixels verificados individualmente célula a célula contra a referência serial canônica [ponto_a_ponto.cpp](file:///home/debrito/Programação/setMandel/src/ponto_a_ponto.cpp)):

| Cenário | Região Real ($\text{Re}$) | Região Imaginária ($\text{Im}$) | $\text{MAX\_ITER}$ | Características e Objetivo |
| :--- | :---: | :---: | :---: | :--- |
| **Full** (Visão Completa) | $[-2.0, 1.0]$ | $[-1.5, 1.5]$ | 1000 | Vista ampla do conjunto. Geometria simétrica em relação ao eixo real. Valida o desdobramento global e o enquadramento espacial. |
| **Zoom** (Seahorse Valley) | $[-0.745144, -0.742144]$ | $[0.130326, 0.133326]$ | 1000 | Região profunda altamente assimétrica e densa em iterações de escape. Valida se o balanceamento dinâmico e o desentrelaçamento cíclico não corrompem detalhes finos. |

---

## ⚙️ 3. Definição dos 5 Algoritmos MPI e Variações

1. **MPI Estático - Blocos Contíguos (`block`)**:
   - Binário: `mpi_estatico`
   - O Rank 0 divide as linhas contiguamente: cada processo recebe uma faixa de tamanho $\approx \lfloor HEIGHT / np \rfloor$.
   - Testado com $np \in \{2, 3, 4\}$.
2. **MPI Estático - Cíclico / Intercalado (`cyclic`)**:
   - Binário: `mpi_estatico`
   - As linhas são distribuídas de forma intercalada (*round-robin*) em lotes de $K$ linhas.
   - Chunks avaliados: $1$, $4$ e $16$ linhas.
   - Processos: $np \in \{2, 3, 4\}$.
3. **MPI Dinâmico - Coordenador Dedicado (`dynamic`)**:
   - Binário: `mpi_dinamico`
   - O Rank 0 atua exclusivamente como despachante de tarefas sob demanda e coletor de resultados. Apenas os trabalhadores ($Ranks \ge 1$) computam pixels.
   - Chunks avaliados: $1$, $4$ e $16$ linhas.
   - Processos: $np \in \{2, 3, 4\}$.
4. **MPI Dinâmico - Coordenador Participativo (`dynamic_part`)**:
   - Binário: `mpi_dinamico_participativo`
   - O Rank 0 despacha tarefas para os trabalhadores, mas nos intervalos ociosos também processa fatias de linhas para si mesmo, maximizando o aproveitamento da CPU.
   - Chunks avaliados: $1$, $4$ e $16$ linhas.
   - Processos: $np \in \{2, 3, 4\}$.
5. **MPI Dinâmico - Meia-Linha (`half-row`)**:
   - Binário: `mpi_dinamico_meia_linha`
   - Decomposição em granularidade fina: cada linha horizontal é dividida em duas metades (metade esquerda $0 \dots W/2$ e metade direita $W/2 \dots W$).
   - Chunks avaliados: $1$, $2$ e $8$ meias-linhas.
   - Processos: $np \in \{2, 3, 4\}$.

---

## 📊 4. Tabela Completa dos 78 Casos de Teste

### Parte A: Cenário 1 — Visão Completa (Testes 01 a 39)

| ID | Cenário | Algoritmo MPI | Schedule | Chunk Size | Processos ($np$) | Dimensões | Resultado |
| :-: | :---: | :--- | :---: | :-: | :-: | :-: | :---: |
| **01** | Full | MPI Estático (Block) | `block` | — | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **02** | Full | MPI Estático (Block) | `block` | — | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **03** | Full | MPI Estático (Block) | `block` | — | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **04** | Full | MPI Estático (Cyclic) | `cyclic` | 1 linha | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **05** | Full | MPI Estático (Cyclic) | `cyclic` | 1 linha | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **06** | Full | MPI Estático (Cyclic) | `cyclic` | 1 linha | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **07** | Full | MPI Estático (Cyclic) | `cyclic` | 4 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **08** | Full | MPI Estático (Cyclic) | `cyclic` | 4 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **09** | Full | MPI Estático (Cyclic) | `cyclic` | 4 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **10** | Full | MPI Estático (Cyclic) | `cyclic` | 16 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **11** | Full | MPI Estático (Cyclic) | `cyclic` | 16 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **12** | Full | MPI Estático (Cyclic) | `cyclic` | 16 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **13** | Full | MPI Dinâmico Dedicado | `dynamic` | 1 linha | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **14** | Full | MPI Dinâmico Dedicado | `dynamic` | 1 linha | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **15** | Full | MPI Dinâmico Dedicado | `dynamic` | 1 linha | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **16** | Full | MPI Dinâmico Dedicado | `dynamic` | 4 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **17** | Full | MPI Dinâmico Dedicado | `dynamic` | 4 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **18** | Full | MPI Dinâmico Dedicado | `dynamic` | 4 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **19** | Full | MPI Dinâmico Dedicado | `dynamic` | 16 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **20** | Full | MPI Dinâmico Dedicado | `dynamic` | 16 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **21** | Full | MPI Dinâmico Dedicado | `dynamic` | 16 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **22** | Full | MPI Dinâmico Participativo | `dynamic_part` | 1 linha | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **23** | Full | MPI Dinâmico Participativo | `dynamic_part` | 1 linha | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **24** | Full | MPI Dinâmico Participativo | `dynamic_part` | 1 linha | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **25** | Full | MPI Dinâmico Participativo | `dynamic_part` | 4 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **26** | Full | MPI Dinâmico Participativo | `dynamic_part` | 4 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **27** | Full | MPI Dinâmico Participativo | `dynamic_part` | 4 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **28** | Full | MPI Dinâmico Participativo | `dynamic_part` | 16 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **29** | Full | MPI Dinâmico Participativo | `dynamic_part` | 16 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **30** | Full | MPI Dinâmico Participativo | `dynamic_part` | 16 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **31** | Full | MPI Dinâmico Meia-Linha | `half-row` | 1 meia-linha | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **32** | Full | MPI Dinâmico Meia-Linha | `half-row` | 1 meia-linha | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **33** | Full | MPI Dinâmico Meia-Linha | `half-row` | 1 meia-linha | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **34** | Full | MPI Dinâmico Meia-Linha | `half-row` | 2 meias-linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **35** | Full | MPI Dinâmico Meia-Linha | `half-row` | 2 meias-linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **36** | Full | MPI Dinâmico Meia-Linha | `half-row` | 2 meias-linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **37** | Full | MPI Dinâmico Meia-Linha | `half-row` | 8 meias-linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **38** | Full | MPI Dinâmico Meia-Linha | `half-row` | 8 meias-linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **39** | Full | MPI Dinâmico Meia-Linha | `half-row` | 8 meias-linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |

---

### Parte B: Cenário 2 — Zoom no Seahorse Valley (Testes 40 a 78)

| ID | Cenário | Algoritmo MPI | Schedule | Chunk Size | Processos ($np$) | Dimensões | Resultado |
| :-: | :---: | :--- | :---: | :-: | :-: | :-: | :---: |
| **40** | Zoom | MPI Estático (Block) | `block` | — | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **41** | Zoom | MPI Estático (Block) | `block` | — | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **42** | Zoom | MPI Estático (Block) | `block` | — | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **43** | Zoom | MPI Estático (Cyclic) | `cyclic` | 1 linha | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **44** | Zoom | MPI Estático (Cyclic) | `cyclic` | 1 linha | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **45** | Zoom | MPI Estático (Cyclic) | `cyclic` | 1 linha | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **46** | Zoom | MPI Estático (Cyclic) | `cyclic` | 4 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **47** | Zoom | MPI Estático (Cyclic) | `cyclic` | 4 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **48** | Zoom | MPI Estático (Cyclic) | `cyclic` | 4 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **49** | Zoom | MPI Estático (Cyclic) | `cyclic` | 16 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **50** | Zoom | MPI Estático (Cyclic) | `cyclic` | 16 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **51** | Zoom | MPI Estático (Cyclic) | `cyclic` | 16 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **52** | Zoom | MPI Dinâmico Dedicado | `dynamic` | 1 linha | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **53** | Zoom | MPI Dinâmico Dedicado | `dynamic` | 1 linha | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **54** | Zoom | MPI Dinâmico Dedicado | `dynamic` | 1 linha | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **55** | Zoom | MPI Dinâmico Dedicado | `dynamic` | 4 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **56** | Zoom | MPI Dinâmico Dedicado | `dynamic` | 4 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **57** | Zoom | MPI Dinâmico Dedicado | `dynamic` | 4 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **58** | Zoom | MPI Dinâmico Dedicado | `dynamic` | 16 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **59** | Zoom | MPI Dinâmico Dedicado | `dynamic` | 16 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **60** | Zoom | MPI Dinâmico Dedicado | `dynamic` | 16 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **61** | Zoom | MPI Dinâmico Participativo | `dynamic_part` | 1 linha | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **62** | Zoom | MPI Dinâmico Participativo | `dynamic_part` | 1 linha | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **63** | Zoom | MPI Dinâmico Participativo | `dynamic_part` | 1 linha | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **64** | Zoom | MPI Dinâmico Participativo | `dynamic_part` | 4 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **65** | Zoom | MPI Dinâmico Participativo | `dynamic_part` | 4 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **66** | Zoom | MPI Dinâmico Participativo | `dynamic_part` | 4 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **67** | Zoom | MPI Dinâmico Participativo | `dynamic_part` | 16 linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **68** | Zoom | MPI Dinâmico Participativo | `dynamic_part` | 16 linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **69** | Zoom | MPI Dinâmico Participativo | `dynamic_part` | 16 linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **70** | Zoom | MPI Dinâmico Meia-Linha | `half-row` | 1 meia-linha | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **71** | Zoom | MPI Dinâmico Meia-Linha | `half-row` | 1 meia-linha | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **72** | Zoom | MPI Dinâmico Meia-Linha | `half-row` | 1 meia-linha | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **73** | Zoom | MPI Dinâmico Meia-Linha | `half-row` | 2 meias-linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **74** | Zoom | MPI Dinâmico Meia-Linha | `half-row` | 2 meias-linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **75** | Zoom | MPI Dinâmico Meia-Linha | `half-row` | 2 meias-linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **76** | Zoom | MPI Dinâmico Meia-Linha | `half-row` | 8 meias-linhas | 2 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **77** | Zoom | MPI Dinâmico Meia-Linha | `half-row` | 8 meias-linhas | 3 | $512 \times 512$ | ✅ 100% Exato (0 difs) |
| **78** | Zoom | MPI Dinâmico Meia-Linha | `half-row` | 8 meias-linhas | 4 | $512 \times 512$ | ✅ 100% Exato (0 difs) |

---

## 🎯 5. Principais Conclusões da Validação

1. **Correspondência Exata Global ($78/78 = 100\%$)**:
   - Em todas as 78 configurações, o utilitário nativo [comparador.cpp](file:///home/debrito/Programação/setMandel/src/comparador.cpp) reportou **0 discrepâncias** em relação à matriz de referência serial [ponto_a_ponto.cpp](file:///home/debrito/Programação/setMandel/src/ponto_a_ponto.cpp).
2. **Reconstituição Perfeita de Matriz no Rank 0**:
   - Os testes comprovam que o desentrelaçamento de dados no `MPI_Gatherv` (MPI Cíclico) e as mensagens ponto a ponto assíncronas (MPI Dinâmico Dedicado, Participativo e Meia-Linha) reconstituíram a ordem estrita *row-major* das linhas sem qualquer deslocamento ou perda de pixels.
3. **Robustez para Processos Ímpares ($np = 3$)**:
   - O teste com $np=3$ foi fundamental para validar a matemática de divisão de linhas e chunks com sobras ($512 \pmod 3 = 2$). Nenhum processo travou por sobreposição ou ausência de tarefas.
