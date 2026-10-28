#!/usr/bin/env python3
"""
==============================================================================
Script de Automação de Benchmarks MPI para o Relatório 2 (setMandel)
Disciplina: DEC107 — Processamento Paralelo (UESC)
==============================================================================
Executa baterias de testes para as versões MPI:
  - mpi_estatico (Estratégias: 'block' e 'cyclic')
  - mpi_dinamico (Estratégia: Coordenador-Trabalhador com variação de chunk)
Registra automaticamente no CSV 'data/dateTimeExecution.csv'.
"""

import os
import sys
import shutil
import subprocess
import time
import argparse
import csv

SCENARIOS = {
    "full": {
        "name": "Visao Completa",
        "RE_MIN": -2.0,
        "RE_MAX": 1.0,
        "IM_MIN": -1.5,
        "IM_MAX": 1.5,
        "MAX_ITER": 1000,
    },
    "zoom": {
        "name": "Zoom (Seahorse Valley)",
        "RE_MIN": -0.745144,
        "RE_MAX": -0.742144,
        "IM_MIN": 0.130326,
        "IM_MAX": 0.133326,
        "MAX_ITER": 5000,
    }
}

def get_root_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir

def check_mpi_available():
    return shutil.which("mpirun") is not None or shutil.which("mpiexec") is not None

def write_in_txt(params, filepath="in.txt"):
    with open(filepath, "w") as f:
        for k, v in params.items():
            f.write(f"{k}={v}\n")

def clean_out_folder(out_dir="out", threshold=6):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        return
    files = [f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
    if len(files) >= threshold:
        for f in files:
            try:
                os.remove(os.path.join(out_dir, f))
            except Exception:
                pass

def run_mpi_test(exe_name, num_procs, root_dir, timeout=600):
    mpi_exec = shutil.which("mpirun") or shutil.which("mpiexec") or "mpirun"
    exe_path = os.path.join(root_dir, exe_name)
    if not os.path.isfile(exe_path):
        exe_path += ".exe"
    
    if not os.path.isfile(exe_path):
        print(f"❌ Binário '{exe_name}' não encontrado. Compile com 'python3 scripts/compile.py'.")
        return None

    # Flags MPI seguras para execução em máquina local (evita travas em containers/root se necessário)
    cmd = [mpi_exec, "-np", str(num_procs)]
    if sys.platform != "win32":
        # Opções amigáveis do OpenMPI
        cmd.extend(["--oversubscribe"])
    cmd.append(f"./{exe_name}")

    env = os.environ.copy()
    env.pop("DISPLAY", None)

    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=root_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        elapsed_wall = time.time() - start_time
        if proc.returncode != 0:
            print(f"\n   ❌ Erro ao executar: {' '.join(cmd)}")
            print(f"      Stderr: {proc.stderr.strip()[:300]}")
            return None
        return elapsed_wall
    except subprocess.TimeoutExpired:
        print(f"\n   ⏳ TIMEOUT ({timeout}s) na execução de {' '.join(cmd)}")
        return None
    except Exception as e:
        print(f"\n   ❌ Exceção: {e}")
        return None

def build_test_suite(resolutions, scenarios, procs_list, mode="all", chunks=[1, 2, 4]):
    tests = []
    for sc_key in scenarios:
        sc = SCENARIOS[sc_key]
        for res in resolutions:
            # 1. MPI Estático: Block e Cyclic (com variação de chunk em cyclic)
            if mode in ["all", "estatico", "static"]:
                for np in procs_list:
                    # Bloco contíguo
                    tests.append({
                        "exe": "mpi_estatico",
                        "strategy": "Estatico (block)",
                        "SCHEDULE": "block",
                        "CHUNK_SIZE": 0,
                        "PROCS": np,
                        "WIDTH": res,
                        "HEIGHT": res,
                        "scenario_key": sc_key,
                        "scenario_name": sc["name"],
                        "MAX_ITER": sc["MAX_ITER"],
                        "RE_MIN": sc["RE_MIN"],
                        "RE_MAX": sc["RE_MAX"],
                        "IM_MIN": sc["IM_MIN"],
                        "IM_MAX": sc["IM_MAX"]
                    })
                    # Cíclico com chunk 1, 2, 4 (block-cyclic)
                    for ch in chunks:
                        tests.append({
                            "exe": "mpi_estatico",
                            "strategy": f"Estatico (cyclic-{ch})",
                            "SCHEDULE": "cyclic",
                            "CHUNK_SIZE": ch,
                            "PROCS": np,
                            "WIDTH": res,
                            "HEIGHT": res,
                            "scenario_key": sc_key,
                            "scenario_name": sc["name"],
                            "MAX_ITER": sc["MAX_ITER"],
                            "RE_MIN": sc["RE_MIN"],
                            "RE_MAX": sc["RE_MAX"],
                            "IM_MIN": sc["IM_MIN"],
                            "IM_MAX": sc["IM_MAX"]
                        })

            # 2. MPI Dinâmico: Coordenador Dedicado
            if mode in ["all", "dinamico", "dynamic"]:
                for ch in chunks:
                    for np in procs_list:
                        tests.append({
                            "exe": "mpi_dinamico",
                            "strategy": f"Dinamico-Ded (chunk={ch})",
                            "SCHEDULE": "dynamic",
                            "CHUNK_SIZE": ch,
                            "PROCS": np,
                            "WIDTH": res,
                            "HEIGHT": res,
                            "scenario_key": sc_key,
                            "scenario_name": sc["name"],
                            "MAX_ITER": sc["MAX_ITER"],
                            "RE_MIN": sc["RE_MIN"],
                            "RE_MAX": sc["RE_MAX"],
                            "IM_MIN": sc["IM_MIN"],
                            "IM_MAX": sc["IM_MAX"]
                        })

            # 3. MPI Dinâmico: Coordenador Participativo (Rank 0 calcula)
            if mode in ["all", "participativo", "part"]:
                for ch in chunks:
                    for np in procs_list:
                        tests.append({
                            "exe": "mpi_dinamico_participativo",
                            "strategy": f"Dinamico-Part (chunk={ch})",
                            "SCHEDULE": "dynamic_part",
                            "CHUNK_SIZE": ch,
                            "PROCS": np,
                            "WIDTH": res,
                            "HEIGHT": res,
                            "scenario_key": sc_key,
                            "scenario_name": sc["name"],
                            "MAX_ITER": sc["MAX_ITER"],
                            "RE_MIN": sc["RE_MIN"],
                            "RE_MAX": sc["RE_MAX"],
                            "IM_MIN": sc["IM_MIN"],
                            "IM_MAX": sc["IM_MAX"]
                        })

            # 4. MPI Dinâmico: Meia-Linha (Sub-linha)
            if mode in ["all", "meia_linha", "half"]:
                for ch in chunks:
                    for np in procs_list:
                        tests.append({
                            "exe": "mpi_dinamico_meia_linha",
                            "strategy": f"Meia-Linha (chunk={ch})",
                            "SCHEDULE": "half-row",
                            "CHUNK_SIZE": ch,
                            "PROCS": np,
                            "WIDTH": res,
                            "HEIGHT": res,
                            "scenario_key": sc_key,
                            "scenario_name": sc["name"],
                            "MAX_ITER": sc["MAX_ITER"],
                            "RE_MIN": sc["RE_MIN"],
                            "RE_MAX": sc["RE_MAX"],
                            "IM_MIN": sc["IM_MIN"],
                            "IM_MAX": sc["IM_MAX"]
                        })
    return tests

def load_existing_executions(csv_path):
    """
    Carrega execuções já salvas no CSV para suportar retomada inteligente (checkpointing).
    Retorna um dicionário: key -> lista de tempos (float).
    key = (Code, Schedule, ChunkSize, Threads, WIDTH, HEIGHT, MAX_ITER)
    """
    runs = {}
    if not os.path.isfile(csv_path):
        return runs
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    code = row.get("Code", "").strip()
                    sched = row.get("Schedule", "").strip()
                    chunk = int(row.get("ChunkSize") or 0)
                    threads = int(row.get("Threads") or 0)
                    w = int(row.get("WIDTH") or 0)
                    h = int(row.get("HEIGHT") or 0)
                    it = int(row.get("MAX_ITER") or 0)
                    t_val = float(row.get("TempoGasto", 0.0))
                    key = (code, sched, chunk, threads, w, h, it)
                    runs.setdefault(key, []).append(t_val)
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"⚠️ Aviso ao ler histórico de execuções: {e}")
    return runs

def execute_battery(tests, reps=3, resume=True):
    root_dir = get_root_dir()
    csv_path = os.path.join(root_dir, "data", "dateTimeExecution.csv")
    if not os.path.exists(csv_path) and os.path.exists(os.path.join(root_dir, "dateTimeExecution.csv")):
        csv_path = os.path.join(root_dir, "dateTimeExecution.csv")

    existing_runs = load_existing_executions(csv_path) if resume else {}

    # Analisa quantos testes já estão completos e quantas repetições faltam
    completed_configs = 0
    pending_reps_total = 0
    for t in tests:
        key = (t["exe"], t["SCHEDULE"], t["CHUNK_SIZE"], t["PROCS"], t["WIDTH"], t["HEIGHT"], t["MAX_ITER"])
        done = len(existing_runs.get(key, []))
        needed = max(0, reps - done)
        if needed == 0:
            completed_configs += 1
        else:
            pending_reps_total += needed

    print(f"\n🚀 Bateria de Testes MPI ({len(tests)} configurações, {reps} repetições por teste)")
    if resume and completed_configs > 0:
        print(f"   🔄 Retomada Inteligente (Resume Ativado):")
        print(f"      - Configurações já 100% concluídas : {completed_configs}/{len(tests)}")
        print(f"      - Configurações com execuções pendentes : {len(tests) - completed_configs}")
        print(f"      - Execuções restantes a rodar : {pending_reps_total} (em vez de {len(tests) * reps})\n")
    else:
        print(f"   Total de execuções planejadas: {len(tests) * reps}\n")

    summary = []
    for idx, t in enumerate(tests, 1):
        key = (t["exe"], t["SCHEDULE"], t["CHUNK_SIZE"], t["PROCS"], t["WIDTH"], t["HEIGHT"], t["MAX_ITER"])
        past_times = existing_runs.get(key, []) if resume else []
        already_done = len(past_times)

        # Se já tiver o número de repetições necessárias no CSV, pula
        if already_done >= reps:
            print(f"⏭️  [{idx:02d}/{len(tests)}] {t['strategy']:<24} | {t['scenario_name']:<22} | {t['WIDTH']}x{t['HEIGHT']} | Procs: {t['PROCS']} | ✅ Já possui {already_done}/{reps} reps no CSV (Pulando!)")
            summary.append({
                "strategy": t["strategy"],
                "scenario": t["scenario_name"],
                "res": f"{t['WIDTH']}x{t['HEIGHT']}",
                "procs": t["PROCS"],
                "avg_time": sum(past_times[:reps]) / min(already_done, reps)
            })
            continue

        reps_to_run = reps - already_done
        print(f"📌 [{idx:02d}/{len(tests)}] {t['strategy']:<24} | {t['scenario_name']:<22} | {t['WIDTH']}x{t['HEIGHT']} | Processos: {t['PROCS']}" +
              (f" (Executando {reps_to_run} rep(s) restante(s) de {reps})" if already_done > 0 else ""))

        # Grava in.txt
        write_in_txt({
            "WIDTH": t["WIDTH"],
            "HEIGHT": t["HEIGHT"],
            "MAX_ITER": t["MAX_ITER"],
            "RE_MIN": t["RE_MIN"],
            "RE_MAX": t["RE_MAX"],
            "IM_MIN": t["IM_MIN"],
            "IM_MAX": t["IM_MAX"],
            "SCHEDULE": t["SCHEDULE"],
            "CHUNK_SIZE": t["CHUNK_SIZE"],
            "THREADS": t["PROCS"],
            "SAVE_IMAGE": 0
        }, filepath=os.path.join(root_dir, "in.txt"))

        new_times = []
        start_rep = already_done + 1
        for r in range(start_rep, reps + 1):
            print(f"   -> Repeticao {r}/{reps}... ", end="", flush=True)
            elapsed = run_mpi_test(t["exe"], t["PROCS"], root_dir)
            if elapsed is not None:
                new_times.append(elapsed)
                print(f"OK ({elapsed:.2f}s)")
            else:
                print("FALHOU")
            clean_out_folder(os.path.join(root_dir, "out"), threshold=6)

        all_times = past_times + new_times
        if all_times:
            avg_t = sum(all_times) / len(all_times)
            summary.append({
                "strategy": t["strategy"],
                "scenario": t["scenario_name"],
                "res": f"{t['WIDTH']}x{t['HEIGHT']}",
                "procs": t["PROCS"],
                "avg_time": avg_t
            })

    print("\n" + "="*80)
    print("  RESUMO DAS EXECUÇÕES MPI CONCLUÍDAS")
    print("="*80)
    print(f"{'Estratégia':<24} | {'Cenário':<22} | {'Resolução':<10} | {'Procs':<6} | {'Tempo Médio (s)':<15}")
    print("-" * 80)
    for s in summary:
        print(f"{s['strategy']:<24} | {s['scenario']:<22} | {s['res']:<10} | {s['procs']:<6} | {s['avg_time']:.3f}s")
    print("="*80)
    print("✅ Todos os resultados foram salvos em 'data/dateTimeExecution.csv'.\n")

def main():
    parser = argparse.ArgumentParser(description="Automação de Benchmarks MPI para Mandelbrot (Relatório 2)")
    parser.add_argument("--mode", choices=["all", "estatico", "dinamico", "participativo", "meia_linha"], default=None,
                        help="Quais versões MPI rodar ('estatico', 'dinamico', 'participativo', 'meia_linha' ou 'all')")
    parser.add_argument("--procs", type=str, default="4",
                        help="Lista de processos separados por vírgula (padrão: 4, ou ex: 1,2,4)")
    parser.add_argument("--res", type=str, default="4096",
                        help="Resoluções separadas por vírgula (ex: 4096,8192 ou 4096,8192,16384)")
    parser.add_argument("--all-res", action="store_true",
                        help="Executar todas as 3 resoluções do benchmark: 4096, 8192 e 16384")
    parser.add_argument("--scenarios", type=str, default="full,zoom",
                        help="Cenários ('full', 'zoom' ou 'full,zoom')")
    parser.add_argument("--reps", type=int, default=5,
                        help="Número de repetições por teste (padrão: 5)")
    parser.add_argument("--chunks", type=str, default="1,2,4",
                        help="Tamanhos de chunk para os testes (padrão: 1,2,4)")
    parser.add_argument("--rapido", action="store_true",
                        help="Modo rápido: 1 repetição, resolução 4096, cenários full")
    parser.add_argument("--no-resume", action="store_true",
                        help="Desativar retomada inteligente (forçar reexecução de testes já presentes no CSV)")

    args = parser.parse_args()

    if not check_mpi_available():
        print("❌ 'mpirun' ou 'mpiexec' não foi encontrado no sistema.")
        print("   Por favor, instale o pacote OpenMPI:")
        print("   - Linux (Ubuntu/Debian): sudo apt update && sudo apt install -y openmpi-bin libopenmpi-dev")
        print("   - Windows (MSYS2): pacman -S mingw-w64-x86_64-msmpi")
        sys.exit(1)

    # Resolução padrão ou todas
    selected_res = "4096,8192,16384" if args.all_res else args.res

    # Menu interativo se nenhum argumento principal foi fornecido
    mode = args.mode
    if mode is None and not args.rapido:
        print("=" * 70)
        print("   BENCHMARK MANDELBROT - SUÍTE MPI (RELATÓRIO 2)")
        print("=" * 70)
        print("  [1] 🚀 RODAR BATERIA COMPLETA (Todos os Códigos com Chunks 1, 2 e 4)")
        print("  [2] 📦 Rodar apenas MPI ESTÁTICO (Blocos Contíguos vs Cíclico 1, 2, 4)")
        print("  [3] ⚡ Rodar apenas MPI DINÂMICO DEDICADO (Chunks 1, 2, 4)")
        print("  [4] 🤝 Rodar apenas MPI DINÂMICO PARTICIPATIVO (Chunks 1, 2, 4)")
        print("  [5] ✂️  Rodar apenas MPI DINÂMICO MEIA-LINHA (Chunks 1, 2, 4)")
        print("  [6] 🧪 Teste Rápido de Verificação (4096x4096, 4 processos, 1 rep)")
        print("  [0] Sair")
        print("=" * 70)
        try:
            choice = input("Digite a opção desejada [1-6]: ").strip()
        except EOFError:
            choice = "1"

        if choice == "1":
            mode = "all"
        elif choice == "2":
            mode = "estatico"
        elif choice == "3":
            mode = "dinamico"
        elif choice == "4":
            mode = "participativo"
        elif choice == "5":
            mode = "meia_linha"
        elif choice == "6":
            args.rapido = True
            mode = "all"
        else:
            print("Encerrando.")
            return

        # Pergunta da resolução no modo interativo (caso não tenha passado flag)
        if not args.rapido and not args.all_res and args.res == "4096":
            print("\n  📐 Escolha as resoluções da imagem:")
            print("     [1] Apenas 4096 (Padrão, mais rápido)")
            print("     [2] 4096 e 8192 (Intermediário)")
            print("     [3] 4096, 8192 e 16384 (Completo: inclui testes 8K e 16K)")
            try:
                r_choice = input("     Opção [1-3, padrão 1]: ").strip()
            except EOFError:
                r_choice = "1"

            if r_choice == "2":
                selected_res = "4096,8192"
            elif r_choice == "3":
                selected_res = "4096,8192,16384"
            else:
                selected_res = "4096"

    if args.rapido:
        procs_list = [1, 2, 4]
        res_list = [4096]
        sc_list = ["full"]
        reps = 1
        chunks_list = [1]
        mode = "all" if mode is None else mode
    else:
        procs_list = [int(p.strip()) for p in args.procs.split(",") if p.strip()]
        res_list = [int(r.strip()) for r in selected_res.split(",") if r.strip()]
        sc_list = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        chunks_list = [int(c.strip()) for c in args.chunks.split(",") if c.strip()]
        reps = args.reps

    tests = build_test_suite(res_list, sc_list, procs_list, mode=mode, chunks=chunks_list)
    execute_battery(tests, reps=reps, resume=not args.no_resume)

if __name__ == "__main__":
    main()
